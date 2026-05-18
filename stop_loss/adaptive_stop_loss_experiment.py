"""
深度条件尾部风险建模与交易生存预测：自适应止损控制实验
论文核心组件：条件分位数预测、EVT尾部建模、神经生存分析

依赖：pandas, numpy, torch, scipy, sklearn, matplotlib, tqdm
"""

import os
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ===================== 配置 =====================
class Config:
    # 数据路径（请根据实际修改）
    signal_dir = "./QuantData/signals/MACD_KDJ"  # 信号文件夹
    ohlcv_dir = "./QuantData/ohlcv"  # 行情文件夹
    results_dir = "./stop_loss_results"
    plot_dir = "./stop_loss_results/figures"

    # 股票列表
    stock_codes = [
        "002311", "002493", "600588", "002049", "000977", "000617", "601111", "600233", "600115", "000661",
        "601006", "002475", "000876", "601600", "002179", "601818", "601117", "600438", "600031", "600089",
        "000568", "002304", "002050", "601318", "600111", "600426", "601618", "600176", "600893", "600104"
    ]
    # 市场：A 或 US
    market = "US"

    # 通用窗口参数
    window_size = 160  # 回溯窗口长度
    future_T = 60  # 预测窗口（最大持有K线数）
    purge_gap = 5  # 防止信息泄露的间隔

    # 标注参数
    base_stop_pct = 0.05  # 用于生成生存标签的固定止损百分比（5%）

    # EVT 参数
    evt_threshold_quantile = 0.9  # 确定阈值 u 的训练集分位数
    evt_tail_fraction = None  # 若为None，则自动计算

    # 分位数列表
    quantile_taus = [0.5, 0.8, 0.9, 0.95]

    # 生存分析参数
    n_intervals = 60  # 离散时间间隔数（通常与 future_T 一致）

    # 模型架构
    encoder_type = 'BiLSTM'
    d_model = 64
    lstm_layers = 2
    dropout = 0.2

    # 训练参数
    batch_size = 32
    lr = 1e-4
    epochs = 200
    patience = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 损失权重
    weight_quantile = 1.0
    weight_evt = 0.3  # 适度权重，不会压制分位数
    weight_survival = 0.1  # 较小权重，辅助作用

    # 回测参数
    max_hold_bars = 60  # 最大持有K线数
    cost_bps_list = [0, 5, 10]  # 交易成本（bps）

    # 自适应止损参数（从分位数选取）
    stop_loss_alpha = 0.8  # 使用 95% 分位数作为止损距离
    # 对比基线
    baseline_stop_methods = [
        'fixed_2pct',  # 固定 2% 止损
        'fixed_5pct',  # 固定 5% 止损
        'atr_2',  # 2倍 ATR 止损
        'atr_3',  # 3倍 ATR 止损
        'trailing_2pct',  # 2% 移动止损
        'mae_regression',  # MAE 回归 LSTM（点估计止损）
        'our_full'  # ← 新增这一行
    ]

    # 多随机种子
    seeds = [11, 22, 33, 42, 55]
    single_seed = 42  # 若 run_multi_seed=False，使用此种子

    run_multi_seed = False
    save_outputs = True
    save_plots = True
    plot_dpi = 150

    # 两阶段训练配置
    train_evt_separately = True  # 是否单独训练 EVT 头
    train_survival_separately = True  # 是否单独训练生存头
    evt_epochs = 50
    survival_epochs = 50
    evt_lr = 1e-4
    survival_lr = 1e-4


cfg = Config()


# ===================== 工具函数 =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ===================== 数据加载与特征工程 =====================
def load_and_prepare(signal_path, ohlcv_path):
    """加载数据并计算技术特征，返回 df 和特征列名"""
    signals = pd.read_csv(signal_path, parse_dates=['time'])
    ohlcv = pd.read_csv(ohlcv_path, parse_dates=['time'])
    df = ohlcv.set_index('time').sort_index()

    # 基本收益率
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_returns'] = df['returns'].abs()

    # 移动平均线
    for w in [5, 10, 20, 60]:
        df[f'ma_{w}'] = df['close'].rolling(w).mean()
        df[f'vol_ma_{w}'] = df['volume'].rolling(w).mean()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # KDJ 简化版
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    rsv = (df['close'] - low_14) / (high_14 - low_14) * 100
    df['k'] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df['d'] = df['k'].ewm(alpha=1 / 3, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # ATR
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 变化率
    df['pct_chg'] = df['close'].pct_change()
    df['vol_chg'] = df['volume'].pct_change()

    # 额外特征：波动率锥、偏度等（可增强尾部感知）
    df['vol_20'] = df['returns'].rolling(20).std()
    df['skew_20'] = df['returns'].rolling(20).skew()

    df['atr_pct'] = df['atr'] / df['close']
    df['vol_regime'] = (
        df['vol_20']
        .rolling(252)
        .rank(pct=True)
    )
    df['trend_strength'] = (
            (df['ma_20'] - df['ma_60']) / df['ma_60']
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'returns', 'abs_returns',
                    'ma_5', 'ma_10', 'ma_20', 'ma_60',
                    'vol_ma_5', 'vol_ma_10', 'vol_ma_20', 'vol_ma_60',
                    'macd', 'macd_signal', 'macd_hist',
                    'rsi', 'k', 'd', 'j', 'atr',
                    'pct_chg', 'vol_chg', 'vol_20', 'skew_20',
                    'atr_pct',
                    'vol_regime',
                    'trend_strength']
    df = df[feature_cols].copy()

    # 合并信号
    signals = signals.set_index('time').sort_index()
    df = df.join(signals[['price', 'signal']], how='left')
    df['signal'] = df['signal'].fillna('none')

    return df, feature_cols


# ===================== 标注函数 =====================
def compute_labels(df, signal_indices, cfg):
    """
    对每个 buy 信号计算：
    - mae: 未来 future_T 内的最大不利偏移 (百分比)
    - event: 是否触发了固定止损 (1 为触发，0 为删失)
    - time_to_stop: 触发时的 K 线序号（从1开始），若删失则为 future_T
    """
    labels = {}
    n = len(df)

    for i in signal_indices:
        if i >= n:
            continue
        if df['signal'].iloc[i] != 'buy':
            continue
        entry_price = df['close'].iloc[i]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        end = min(n, i + cfg.future_T + 1)
        if end <= i + 1:
            continue

        # 计算 MAE
        lows = df['low'].iloc[i + 1:end].values

        smooth_lows = (
            pd.Series(lows)
            .rolling(3, min_periods=1)
            .mean()
            .values
        )

        mae = np.max((entry_price - smooth_lows) / entry_price)

        # 固定止损触发计算
        stop_price = entry_price * (1 - cfg.base_stop_pct)
        event = 0
        tts = cfg.future_T  # 默认删失时生存时间为最大长度
        for k, low in enumerate(lows, start=1):
            if low <= stop_price:
                event = 1
                tts = k
                break

        atr_ratio = df['atr'].iloc[i] / entry_price
        mae_norm = mae / max(atr_ratio, 1e-4)

        labels[i] = {
            'mae': mae_norm,
            'event': event,
            'time_to_stop': tts
        }

    return labels


# ===================== 数据集 =====================
class StopLossDataset(Dataset):
    """
    返回：
    - x_win: 回溯窗口特征 (L, d)
    - mae: 目标 MAE 值
    - event: 生存事件指示 (0 或 1)
    - tts: 生存时间 (1..future_T)
    """

    def __init__(self, df, feature_cols, cfg, scaler=None,
                 signal_indices=None, labels_dict=None):
        self.df = df
        self.features = df[feature_cols].values.astype(np.float32)
        self.window = cfg.window_size
        self.cfg = cfg

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

        # 仅保留有有效标签的 buy 信号
        self.indices = []
        self.mae_list = []
        self.event_list = []
        self.tts_list = []

        for idx in signal_indices:
            if idx < self.window:
                continue
            if idx not in labels_dict:
                continue
            lab = labels_dict[idx]
            if np.isnan(lab['mae']):
                continue
            self.indices.append(idx)
            self.mae_list.append(lab['mae'])
            self.event_list.append(lab['event'])
            self.tts_list.append(lab['time_to_stop'])

        self.mae_list = np.array(self.mae_list, dtype=np.float32)
        self.event_list = np.array(self.event_list, dtype=np.float32)
        self.tts_list = np.array(self.tts_list, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_win = self.features[i - self.window:i]  # (L, d)
        mae = self.mae_list[idx]
        event = self.event_list[idx]
        tts = self.tts_list[idx]
        return (
            torch.tensor(x_win, dtype=torch.float32),
            torch.tensor(mae, dtype=torch.float32),
            torch.tensor(event, dtype=torch.float32),
            torch.tensor(tts, dtype=torch.long)
        )


# ===================== 模型定义 =====================
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, d_model // 2, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.proj(out)


class StopLossPredictor(nn.Module):
    """
    多任务输出：
    - quantiles: 多个条件分位数 τ 的预测值
    - evt_params: GPD 参数 (ξ, logβ)
    - hazard_logits: 每个离散时间间隔的危险 logit
    """

    def __init__(self, input_dim, d_model, n_quantiles, n_intervals, dropout=0.2):
        super().__init__()
        self.encoder = BiLSTMEncoder(input_dim, d_model, dropout=dropout)
        self.n_quantiles = n_quantiles
        self.n_intervals = n_intervals

        # 分位数头（确保输出非负）
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                          nn.Linear(d_model // 2, 1), nn.Softplus())
            for _ in range(n_quantiles)
        ])

        # EVT 参数头
        self.evt_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)  # 输出 xi, log_beta
        )

        # 生存危险率头
        self.surv_head = nn.Linear(d_model, n_intervals)  # 输出 logits

    def forward(self, x):
        h_seq = self.encoder(x)
        h = h_seq[:, -1, :]  # 取最后时间步

        quantiles = [head(h).squeeze(-1) for head in self.quantile_heads]  # list of (batch,)
        quantiles = torch.stack(quantiles, dim=1)  # (batch, n_quantiles)

        evt_params = self.evt_head(h)  # (batch, 2)
        xi = evt_params[:, 0]
        log_beta = evt_params[:, 1]  # 确保 beta > 0

        hazard_logits = self.surv_head(h)  # (batch, n_intervals)

        return quantiles, xi, log_beta, hazard_logits


# ===================== 损失函数 =====================
def quantile_loss(pred, target, taus, reduce=True):
    """Pinball loss for multiple quantiles."""
    # pred: (batch, n_quantiles), target: (batch,)
    target = target.unsqueeze(1)  # (batch, 1)
    errors = target - pred  # (batch, n_quantiles)
    loss = torch.max((taus - 1) * errors, taus * errors)
    if reduce:
        return loss.mean()
    return loss


def evt_gpd_nll(mae, u, xi, log_beta):
    """GPD 负对数似然，仅对超过阈值的超额 y = mae - u 计算"""
    exceed = mae > u
    if exceed.sum() == 0:
        return torch.tensor(0.0, device=mae.device, requires_grad=True)

    y = (mae[exceed] - u).clamp(min=1e-8)
    beta = torch.exp(log_beta[exceed]).clamp(min=1e-8)
    xi = xi[exceed]

    # 处理 xi 接近 0 的情况（指数分布）
    small_xi = torch.abs(xi) < 1e-6
    nll = torch.zeros_like(y)

    # 一般情况：ξ != 0
    mask = ~small_xi
    if mask.any():
        z = 1 + xi[mask] * y[mask] / beta[mask]
        z = z.clamp(min=1e-8)
        nll[mask] = torch.log(beta[mask]) + (1.0 / xi[mask] + 1.0) * torch.log(z)

    # 指数分布极限
    if small_xi.any():
        nll[small_xi] = torch.log(beta[small_xi]) + y[small_xi] / beta[small_xi]

    return nll.mean()


def survival_bce_loss(hazard_logits, tts, events, n_intervals):
    """
    离散时间生存损失：每个时间步做二分类，
    直到 tts 步，事件步目标为 1，非事件步为 0，删失样本全为 0。
    """
    batch_size = hazard_logits.size(0)
    device = hazard_logits.device
    losses = []

    for i in range(batch_size):
        t = tts[i].item() - 1  # 转为 0-based 索引
        e = events[i].item()
        if t >= n_intervals:
            t = n_intervals - 1
        # 只考虑 0..t 步
        logits = hazard_logits[i, :t + 1]
        targets = torch.zeros(t + 1, device=device)
        if e == 1:  # 事件
            targets[-1] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        losses.append(loss)

    return torch.stack(losses).mean()


# ===================== 训练辅助 =====================
def train_epoch(model, dataloader, optimizer, cfg, u_threshold):
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    taus = torch.tensor(cfg.quantile_taus, device=cfg.device)

    for x, mae, event, tts in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(cfg.device)
        mae = mae.to(cfg.device)
        event = event.to(cfg.device)
        tts = tts.to(cfg.device)

        quantiles, xi, log_beta, hazard_logits = model(x)

        loss_q = quantile_loss(quantiles, mae, taus)
        crossing_penalty = (
            F.relu(quantiles[:, :-1] - quantiles[:, 1:])
        ).mean()

        loss_q = loss_q + 5.0 * crossing_penalty
        loss_evt = evt_gpd_nll(mae, u_threshold, xi, log_beta)
        loss_surv = survival_bce_loss(hazard_logits, tts, event, cfg.n_intervals)

        loss = cfg.weight_quantile * loss_q + \
               cfg.weight_evt * loss_evt + \
               cfg.weight_survival * loss_surv

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


def train_evt_head_only(model, dataloader, optimizer, cfg, u_threshold):
    """冻结编码器，只训练 EVT 头"""
    model.eval()  # 编码器保持 eval，但 EVT 头仍可训练
    # 冻结编码器参数
    for param in model.encoder.parameters():
        param.requires_grad = False
    # 解冻 EVT 头
    for param in model.evt_head.parameters():
        param.requires_grad = True
    # 分位数头也冻结
    for head in model.quantile_heads:
        for param in head.parameters():
            param.requires_grad = False
    # 生存头冻结
    for param in model.surv_head.parameters():
        param.requires_grad = False

    total_loss = 0.0
    n_batches = 0
    for x, mae, _, _ in dataloader:
        x = x.to(cfg.device)
        mae = mae.to(cfg.device)
        optimizer.zero_grad()
        quantiles, xi, log_beta, _ = model(x)
        loss = evt_gpd_nll(mae, u_threshold, xi, log_beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def train_survival_head_only(model, dataloader, optimizer, cfg):
    """冻结编码器，只训练生存头"""
    model.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.evt_head.parameters():
        param.requires_grad = False
    for head in model.quantile_heads:
        for param in head.parameters():
            param.requires_grad = False
    for param in model.surv_head.parameters():
        param.requires_grad = True

    total_loss = 0.0
    n_batches = 0
    for x, mae, event, tts in dataloader:
        x = x.to(cfg.device)
        event = event.to(cfg.device)
        tts = tts.to(cfg.device)
        optimizer.zero_grad()
        _, _, _, hazard_logits = model(x)
        loss = survival_bce_loss(hazard_logits, tts, event, cfg.n_intervals)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def eval_metrics(model, dataloader, cfg, u_threshold):
    """在验证集上计算各项损失及分位数覆盖率"""
    model.eval()
    taus = torch.tensor(cfg.quantile_taus, device=cfg.device)
    total_loss_q = 0.0
    total_loss_evt = 0.0
    total_loss_surv = 0.0
    n_batches = 0

    coverage = {tau: [] for tau in cfg.quantile_taus}

    with torch.no_grad():
        for x, mae, event, tts in dataloader:
            x = x.to(cfg.device)
            mae = mae.to(cfg.device)
            event = event.to(cfg.device)
            tts = tts.to(cfg.device)

            quantiles, xi, log_beta, hazard_logits = model(x)

            loss_q = quantile_loss(quantiles, mae, taus)
            loss_evt = evt_gpd_nll(mae, u_threshold, xi, log_beta)
            loss_surv = survival_bce_loss(hazard_logits, tts, event, cfg.n_intervals)

            total_loss_q += loss_q.item()
            total_loss_evt += loss_evt.item()
            total_loss_surv += loss_surv.item()
            n_batches += 1

            # 记录分位数覆盖率
            q_np = quantiles.cpu().numpy()
            mae_np = mae.cpu().numpy()
            for j, tau in enumerate(cfg.quantile_taus):
                cov = np.mean(mae_np <= q_np[:, j])
                coverage[tau].append(cov)

    avg_loss_q = total_loss_q / n_batches
    avg_loss_evt = total_loss_evt / n_batches
    avg_loss_surv = total_loss_surv / n_batches
    avg_coverage = {tau: np.mean(v) for tau, v in coverage.items()}

    return avg_loss_q, avg_loss_evt, avg_loss_surv, avg_coverage


# ===================== 回测引擎 =====================
def apply_stop_loss(equity_curve, position, entry_price, stop_price, current_low, current_close):
    """
    检查是否触发止损。
    返回 (new_position, new_capital, hit_stop)
    """
    if position > 0 and current_low <= stop_price:
        # 止损触发，按止损价平仓（或按最低价更保守）
        exit_price = stop_price
        capital = position * exit_price * (1 - 0)  # 成本后续统一扣除
        return 0.0, capital, True
    return position, None, False


def compute_survival_prob(hazard_logits):
    """将危险率 logits 转为生存概率（离散时间）"""
    hazards = torch.sigmoid(hazard_logits)  # (n_intervals,)
    surv = torch.cumprod(1 - hazards, dim=0)
    return surv.cpu().numpy()


def backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                       stop_method, stop_params, signal_locs,
                       start_loc, end_loc, cost_bps, code):
    """
    执行带止损策略的回测。
    stop_method 可为：
    - 'fixed': 固定百分比，stop_params={'pct': 0.02}
    - 'atr': ATR倍数，stop_params={'mult': 2.0}
    - 'trailing': 移动止损，stop_params={'pct': 0.02}
    - 'mae_regression': 用 MAE 预测的均值作为止损
    - 'our_model': 使用分位数 Q_alpha 作为止损距离
    """
    cost_rate = cost_bps / 10000.0
    capital = 1.0
    position = 0.0
    equity = [capital]
    records = []
    trade_surv = {}  # entry_loc -> survival probability array

    # 跟踪持仓信息
    entry_price = 0.0
    entry_loc = -1  # 新增，记录入场索引
    highest_close = 0.0  # 用于移动止损
    stop_price = 0.0
    model.eval()

    for i_loc in range(start_loc, end_loc):
        row = df.iloc[i_loc]
        idx = df.index[i_loc]
        close = row['close']
        low = row['low']
        high = row['high']  # 可能用于移动止损

        # 检查止损
        if position > 0:
            # 动态生存预警（仅 our_full 方法）
            if stop_method == 'our_full' and position > 0 and entry_loc in trade_surv:
                days_held = i_loc - entry_loc
                if days_held < len(trade_surv[entry_loc]):
                    surv_now = trade_surv[entry_loc][days_held]
                    # 若生存概率低于 0.3 且当前浮亏已达止损距离的 80%
                    current_loss = (entry_price - close) / entry_price
                    if surv_now < 0.3 and current_loss > 0 and current_loss > (1 - stop_price / entry_price) * 0.8:
                        capital = position * close * (1 - cost_rate)
                        position = 0.0
                        records.append({'time': idx, 'type': 'survival_exit', 'price': close})
                        print(f"[{code}] Survival early exit at {close:.2f}, surv={surv_now:.2f}")
                        # 跳过后续
                        equity.append(capital)
                        continue
            hit_stop = False
            if stop_method in ['fixed', 'atr', 'our_model', 'mae_regression']:
                if low <= stop_price:
                    exit_price = stop_price
                    capital = position * exit_price * (1 - cost_rate)
                    position = 0.0
                    hit_stop = True
                    records.append({'time': idx, 'type': 'stop', 'price': exit_price})
                    # 诊断打印：止损触发
                    print(f"[{code}] STOP hit at {exit_price:.2f} (entry={entry_price:.2f})")
            elif stop_method == 'trailing':
                # 更新最高价和止损价
                if close > highest_close:
                    highest_close = close
                    stop_price = highest_close * (1 - stop_params['pct'])
                if low <= stop_price:
                    exit_price = stop_price
                    capital = position * exit_price * (1 - cost_rate)
                    position = 0.0
                    hit_stop = True
                    records.append({'time': idx, 'type': 'stop', 'price': exit_price})
                    print(f"[{code}] STOP hit (trailing) at {exit_price:.2f} (entry={entry_price:.2f})")

            if not hit_stop and row['signal'] == 'sell':
                # 卖出信号平仓
                capital = position * close * (1 - cost_rate)
                position = 0.0
                records.append({'time': idx, 'type': 'sell_signal', 'price': close})

            # 最大持有期检查（在 i_loc 达到 entry_loc + max_hold_bars 时强制平仓）
            # 此处简化：若持仓超过 max_hold_bars 且未平仓，则收盘强平
            # 这里暂不实现复杂计时，用简单方式：回测中在每日结束后检查，但更稳健的做法是记录入场位置
            # 由于当前简化循环，我们暂且忽略最大持有期强制平仓（可在后续优化）。
        # 买入信号处理
        if i_loc in signal_locs and row['signal'] == 'buy' and position == 0:
            entry_price = close
            entry_loc = i_loc  # 记录入场位置
            # 计算止损价
            if stop_method == 'fixed':
                stop_price = entry_price * (1 - stop_params['pct'])
            elif stop_method == 'atr':
                atr_val = row['atr'] if 'atr' in df.columns else 0.02 * entry_price
                stop_price = entry_price - stop_params['mult'] * atr_val
            elif stop_method == 'mae_regression':
                # 使用回归模型预测的 MAE 均值作为止损距离
                x_win = scaler.transform(df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values)
                x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    quantiles, _, _, _ = model(x_ten)
                    pred_mae = quantiles[0, 0].item()  # 中位数或均值，这里取 tau=0.5
                stop_price = entry_price * (1 - max(pred_mae, 0.001))
            elif stop_method == 'our_model':
                # 纯分位数止损（与之前相同）
                x_win = scaler.transform(df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values)
                x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    quantiles, _, _, _ = model(x_ten)
                    alpha = cfg.stop_loss_alpha
                    taus = np.array(cfg.quantile_taus)
                    q_vals = quantiles[0].cpu().numpy()
                    if alpha in taus:
                        pred_q = float(q_vals[taus == alpha][0])
                    else:
                        pred_q = float(np.interp(alpha, taus, q_vals))
                stop_distance = max(pred_q, 0.002)
                stop_price = entry_price * (1 - stop_distance)
                # 我们的完整模型可能会额外保存 surv_prob 等（这里不存）
            elif stop_method == 'our_full':
                # 分位数止损 + 生存动态预警
                x_win = scaler.transform(df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values)
                x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    quantiles, xi, log_beta, hazard_logits = model(x_ten)
                    alpha = cfg.stop_loss_alpha
                    taus = np.array(cfg.quantile_taus)
                    q_vals = quantiles[0].cpu().numpy()
                    if alpha in taus:
                        pred_q = float(q_vals[taus == alpha][0])
                    else:
                        pred_q = float(np.interp(alpha, taus, q_vals))
                stop_distance = max(pred_q, 0.002)
                stop_price = entry_price * (1 - stop_distance)
                # 保存生存概率序列和当前危险率阈值等参数（在 records 中标记）
                surv_prob = compute_survival_prob(hazard_logits[0])  # 所有时间步的生存概率
                trade_surv[entry_loc] = surv_prob
                # 将 surv_prob 存入 records 或额外变量，以便持仓期间查询
                # 我们这里简单把 surv_prob 附加到 records 里 (可后续在持仓检查中使用)
                # 注意：这里买入记录会和下面统一的 records.append 重复，需要调整
                # 我们采用另一种方式：在买入时存储 surv_prob 到一个字典
                # 建议修改：先计算并存储 surv_prob 到外部字典 survival_info[entry_loc] = surv_prob
                # 然后在持仓循环中查询。为简化，我们在此用注释说明，实际通过全局变量传递。
            elif stop_method == 'trailing':
                highest_close = entry_price
                stop_price = entry_price * (1 - stop_params['pct'])
            else:
                stop_price = 0.0

            if stop_method == 'our_model':
                print(f"[{code}] BUY at {entry_price:.2f}, "
                      f"pred_q (α={alpha:.2f}) = {pred_q:.4f} ({pred_q * 100:.2f}%), "
                      f"stop_price = {stop_price:.2f} "
                      f"(distance = {(1 - stop_price / entry_price) * 100:.2f}%)")

            # 开仓
            position = capital * (1 - cost_rate) / entry_price
            capital = 0.0
            records.append({'time': idx, 'type': 'buy', 'price': entry_price})

        # 计算当前净值
        if position > 0:
            equity.append(position * close)
        else:
            equity.append(capital)

    # 最终若还有持仓，按最后一个收盘价平仓
    if position > 0:
        final_price = df['close'].iloc[end_loc - 1]
        capital = position * final_price * (1 - cost_rate)
        equity[-1] = capital  # 修正最后一天的净值
        position = 0.0

    return equity, records


def backtest_no_stop(df, signal_locs, start_loc, end_loc, cost_bps):
    """无止损的基准回测（仅跟随原始信号）"""
    cost_rate = cost_bps / 10000.0
    capital = 1.0
    position = 0.0
    equity = [capital]

    for i_loc in range(start_loc, end_loc):
        row = df.iloc[i_loc]
        if i_loc in signal_locs and row['signal'] != 'none':
            if row['signal'] == 'buy' and position == 0:
                position = capital * (1 - cost_rate) / row['close']
                capital = 0.0
            elif row['signal'] == 'sell' and position > 0:
                capital = position * row['close'] * (1 - cost_rate)
                position = 0.0
        equity.append(position * row['close'] if position > 0 else capital)

    if position > 0:
        capital = position * df['close'].iloc[end_loc - 1] * (1 - cost_rate)
        equity[-1] = capital
    return equity


# ===================== 指标计算 =====================
def calc_sharpe(equity, periods=252):
    eq = np.array(equity)
    if len(eq) < 3 or np.any(eq[:-1] <= 0):
        return 0.0
    rets = eq[1:] / eq[:-1] - 1
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return 0.0
    std = rets.std()
    if std < 1e-12:
        return 0.0
    return np.sqrt(periods) * rets.mean() / std


def calc_max_drawdown(equity):
    eq = np.array(equity)
    if len(eq) == 0: return 0.0
    peak = np.maximum.accumulate(eq)
    peak[peak < 1e-12] = 1e-12
    dd = (eq - peak) / peak
    return dd.min()


def calc_total_return(equity):
    eq = np.array(equity)
    if len(eq) == 0: return 0.0
    return eq[-1] / eq[0] - 1.0


def calc_expectile_loss(equity, alpha=0.05):
    """简化的期望损失（尾部平均损失）"""
    eq = np.array(equity)
    if len(eq) < 2: return 0.0
    rets = eq[1:] / eq[:-1] - 1
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2: return 0.0
    threshold = np.quantile(rets, 1 - alpha)
    tail = rets[rets <= threshold]
    if len(tail) == 0: return 0.0
    return -tail.mean()


# ===================== 主流程 =====================
def run_one_seed(seed):
    set_seed(seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)

    # 构建路径
    if cfg.market == 'A':
        stock_codes = cfg.stock_codes
        ohlcv_paths = [os.path.join(cfg.ohlcv_dir, f"bar_A_{code}_d.csv") for code in stock_codes]
        signal_paths = [os.path.join(cfg.signal_dir, f"A_{code}_d.csv") for code in stock_codes]
    else:  # US 示例
        stock_codes = [
            "NFLX", "PYPL", "CSCO", "IBM", "BAC", "GS", "C", "T", "PFE", "CAT",
            "F", "AXP", "AIG", "MMM", "LOW", "BBY", "WBA", "CMG", "UAL", "EA",
            "PM", "HPQ", "MNST", "COF", "SYY", "AAPL", "MSFT", "TSLA", "WMT", "KO"
        ]
        ohlcv_paths = [os.path.join(cfg.ohlcv_dir, f"bar_USA_{code}_d.csv") for code in stock_codes]
        signal_paths = [os.path.join(cfg.signal_dir, f"USA_{code}_d.csv") for code in stock_codes]

    # 第一阶段：收集训练数据并拟合全局 scaler
    global_train_features = []
    stock_info = []  # 存储 (code, df, feature_cols, train_indices, val_indices, test_indices, labels_dict)

    for code, sig_path, ohlcv_path in zip(stock_codes, signal_paths, ohlcv_paths):
        print(f"Processing {code}...")
        df, feature_cols = load_and_prepare(sig_path, ohlcv_path)

        # 信号索引
        signal_df = df[df['signal'] != 'none'].sort_index()
        n_sig = len(signal_df)
        train_end = int(n_sig * 0.6)
        val_end = int(n_sig * 0.8)

        train_times = signal_df.index[:train_end]
        val_times = signal_df.index[train_end:val_end]
        test_times = signal_df.index[val_end:]

        train_locs = [df.index.get_loc(t) for t in train_times]
        val_locs = [df.index.get_loc(t) for t in val_times]
        test_locs = [df.index.get_loc(t) for t in test_times]

        # 应用 purge_gap
        if val_locs and train_locs:
            min_val = min(val_locs)
            train_locs = [loc for loc in train_locs if loc + cfg.future_T + cfg.purge_gap < min_val]
        if test_locs and val_locs:
            min_test = min(test_locs)
            val_locs = [loc for loc in val_locs if loc + cfg.future_T + cfg.purge_gap < min_test]

        # 生成标签
        labels = compute_labels(df, train_locs + val_locs + test_locs, cfg)

        # 提取训练期特征用于 scaler
        train_mask = df.index <= max(train_times) if len(train_times) > 0 else False
        if len(train_times) > 0:
            global_train_features.append(df[feature_cols].loc[train_mask].values)

        stock_info.append((code, df, feature_cols, train_locs, val_locs, test_locs, labels))

    if not global_train_features:
        print("No training data.")
        return

    # 拟合全局 scaler
    all_train_data = np.concatenate(global_train_features, axis=0)
    scaler = StandardScaler().fit(all_train_data)
    print("Global scaler fitted.")

    # 第二阶段：构建训练/验证集
    train_datasets, val_datasets = [], []
    test_info = []

    for code, df, feature_cols, train_locs, val_locs, test_locs, labels in stock_info:
        train_ds = StopLossDataset(df, feature_cols, cfg, scaler=scaler,
                                   signal_indices=train_locs, labels_dict=labels)
        val_ds = StopLossDataset(df, feature_cols, cfg, scaler=scaler,
                                 signal_indices=val_locs, labels_dict=labels)
        test_ds = StopLossDataset(df, feature_cols, cfg, scaler=scaler,
                                  signal_indices=test_locs, labels_dict=labels)

        if len(train_ds) > 0:
            train_datasets.append(train_ds)
        if len(val_ds) > 0:
            val_datasets.append(val_ds)
        test_info.append((code, df, feature_cols, test_locs, labels, test_ds))

    if not train_datasets:
        print("No training samples.")
        return

    combined_train = torch.utils.data.ConcatDataset(train_datasets)
    combined_val = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None

    # 使用加权采样平衡事件/删失比例（可选）
    # 此处简化，使用普通 DataLoader
    train_loader = DataLoader(combined_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(combined_val, batch_size=cfg.batch_size, shuffle=False) if combined_val else None

    # 确定 EVT 阈值 u：基于训练集 MAE 的分位数
    all_mae = []
    for ds in train_datasets:
        all_mae.extend(ds.mae_list.tolist())
    u_threshold = np.quantile(all_mae, cfg.evt_threshold_quantile)
    print(f"EVT threshold u = {u_threshold:.4f}")

    # 初始化模型
    model = StopLossPredictor(
        input_dim=len(feature_cols),
        d_model=cfg.d_model,
        n_quantiles=len(cfg.quantile_taus),
        n_intervals=cfg.n_intervals,
        dropout=cfg.dropout
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(cfg.results_dir, f"best_model_seed{seed}.pth")

    # 第一阶段：纯分位数训练（冻结其他头的影响）
    original_weight_evt = cfg.weight_evt
    original_weight_survival = cfg.weight_survival
    cfg.weight_evt = 0.0
    cfg.weight_survival = 0.0

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, cfg, u_threshold)

        if val_loader:
            loss_q, loss_evt, loss_surv, coverages = eval_metrics(model, val_loader, cfg, u_threshold)
            val_loss = loss_q + loss_evt + loss_surv  # 简单和
            print(f"Epoch {epoch + 1:03d}: Train Loss={train_loss:.4f}, "
                  f"Val Loss Q={loss_q:.4f} EVT={loss_evt:.4f} Surv={loss_surv:.4f}, "
                  f"Coverage={ {t: f'{c:.2f}' for t, c in coverages.items()} }")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print("Early stopping triggered.")
                    break
        else:
            print(f"Epoch {epoch + 1:03d}: Train Loss={train_loss:.4f}")
            torch.save(model.state_dict(), best_model_path)

    # 加载第一阶段最佳模型（验证集上表现最好的权重）
    model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
    print("Loaded best encoder from phase 1.")

    cfg.weight_evt = original_weight_evt
    cfg.weight_survival = original_weight_survival

    # ---------- 第二阶段：单独训练 EVT 头 ----------
    if cfg.train_evt_separately and cfg.weight_evt > 0:
        print("\n===== Phase 2: Train EVT head with frozen encoder =====")
        # 重置优化器（仅优化 EVT 头参数）
        optimizer_evt = torch.optim.Adam(model.evt_head.parameters(), lr=cfg.evt_lr)
        best_evt_loss = float('inf')
        for epoch in range(cfg.evt_epochs):
            evt_loss = train_evt_head_only(model, train_loader, optimizer_evt, cfg, u_threshold)
            print(f"EVT Epoch {epoch + 1:03d}, Loss={evt_loss:.4f}")
            if evt_loss < best_evt_loss:
                best_evt_loss = evt_loss
                torch.save(model.state_dict(), best_model_path.replace('.pth', '_evt.pth'))

    # ---------- 第三阶段：单独训练生存头 ----------
    if cfg.train_survival_separately and cfg.weight_survival > 0:
        print("\n===== Phase 3: Train survival head with frozen encoder =====")
        optimizer_surv = torch.optim.Adam(model.surv_head.parameters(), lr=cfg.survival_lr)
        best_surv_loss = float('inf')
        for epoch in range(cfg.survival_epochs):
            surv_loss = train_survival_head_only(model, train_loader, optimizer_surv, cfg)
            print(f"Survival Epoch {epoch + 1:03d}, Loss={surv_loss:.4f}")
            if surv_loss < best_surv_loss:
                best_surv_loss = surv_loss
                torch.save(model.state_dict(), best_model_path.replace('.pth', '_surv.pth'))

    # 第二阶段训练结束后，更新最佳模型路径
    if cfg.train_survival_separately and cfg.weight_survival > 0:
        best_model_path = best_model_path.replace('.pth', '_surv.pth')
    elif cfg.train_evt_separately and cfg.weight_evt > 0:
        best_model_path = best_model_path.replace('.pth', '_evt.pth')

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
    model.eval()

    # ======= 回测与评估 =======
    summary_rows = []
    all_equity_curves = defaultdict(list)  # 用于绘图

    for code, df, feature_cols, test_locs, labels, test_ds in test_info:
        if len(test_locs) == 0:
            continue
        start_loc = max(min(test_locs), cfg.window_size)
        end_loc = len(df)
        test_signal_set = set(test_locs)

        for cost_bps in cfg.cost_bps_list:
            # 1. 无止损（原始信号）
            eq_no_stop = backtest_no_stop(df, test_signal_set, start_loc, end_loc, cost_bps)

            # 2. 我们的模型
            eq_our, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                           'our_model', {}, test_signal_set,
                                           start_loc, end_loc, cost_bps, code)

            # 3. 基线止损方法
            baselines_equity = {}
            for method in cfg.baseline_stop_methods:
                if method == 'fixed_2pct':
                    eq, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                               'fixed', {'pct': 0.02}, test_signal_set,
                                               start_loc, end_loc, cost_bps, code)
                    baselines_equity['fixed_2pct'] = eq
                elif method == 'fixed_5pct':
                    eq, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                               'fixed', {'pct': 0.05}, test_signal_set,
                                               start_loc, end_loc, cost_bps, code)
                    baselines_equity['fixed_5pct'] = eq
                elif method == 'atr_2':
                    eq, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                               'atr', {'mult': 2.0}, test_signal_set,
                                               start_loc, end_loc, cost_bps, code)
                    baselines_equity['atr_2'] = eq
                elif method == 'atr_3':
                    eq, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                               'atr', {'mult': 3.0}, test_signal_set,
                                               start_loc, end_loc, cost_bps, code)
                    baselines_equity['atr_3'] = eq
                elif method == 'trailing_2pct':
                    eq, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                               'trailing', {'pct': 0.02}, test_signal_set,
                                               start_loc, end_loc, cost_bps, code)
                    baselines_equity['trailing_2pct'] = eq
                elif method == 'mae_regression':
                    eq, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                               'mae_regression', {}, test_signal_set,
                                               start_loc, end_loc, cost_bps, code)
                    baselines_equity['mae_regression'] = eq
                elif method == 'our_full':
                    eq, _ = backtest_stop_loss(df, model, scaler, feature_cols, cfg,
                                               'our_full', {}, test_signal_set,
                                               start_loc, end_loc, cost_bps, code)
                    baselines_equity['our_full'] = eq

            # 计算指标
            def record_metrics(name, equity):
                return {
                    'code': code,
                    'cost_bps': cost_bps,
                    'method': name,
                    'sharpe': calc_sharpe(equity),
                    'max_dd': calc_max_drawdown(equity),
                    'total_return': calc_total_return(equity),
                    'expectile_loss': calc_expectile_loss(equity)
                }

            summary_rows.append(record_metrics('no_stop', eq_no_stop))
            summary_rows.append(record_metrics('our_model', eq_our))
            for mname, eq in baselines_equity.items():
                summary_rows.append(record_metrics(mname, eq))

            # 保存曲线数据供后续画图
            all_equity_curves[code].append({
                'cost': cost_bps,
                'no_stop': eq_no_stop,
                'our': eq_our,
                'baselines': baselines_equity
            })

    # 汇总结果
    summary_df = pd.DataFrame(summary_rows)
    if cfg.save_outputs:
        summary_path = os.path.join(cfg.results_dir, f"summary_seed{seed}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")

    # 打印平均结果
    print("\n===== Average Performance by Method (cost=0) =====")
    cost0 = summary_df[summary_df['cost_bps'] == 0].groupby('method').agg(
        sharpe_mean=('sharpe', 'mean'),
        max_dd_mean=('max_dd', 'mean'),
        return_mean=('total_return', 'mean'),
        es_mean=('expectile_loss', 'mean')
    ).reset_index()
    print(cost0.to_string(index=False))

    # ---------- 绘图 ----------
    if cfg.save_plots:
        # 绘制平均权益曲线（各成本）
        for cost in cfg.cost_bps_list:
            plot_equity_curves(all_equity_curves, cfg, cost_bps=cost, seed=seed)
        # 绘制风险指标对比图
        plot_risk_metrics(summary_df, cfg)

    return summary_df


# ===================== 绘图函数 =====================
def plot_equity_curves(all_equity_data, cfg, cost_bps=0, seed=None):
    """
    绘制所有股票的平均累积收益曲线（通过插值对齐到统一步数）。
    """
    if seed is None:
        seed = cfg.single_seed
    os.makedirs(cfg.plot_dir, exist_ok=True)

    methods = ['no_stop', 'our_model', 'our_full', 'fixed_2pct', 'fixed_5pct',
               'atr_2', 'atr_3', 'trailing_2pct', 'mae_regression']
    method_labels = {
        'no_stop': 'No Stop',
        'our_model': 'Our Model (EVT+Survival)',
        'our_full': 'Our Full (Q+Surv)',
        'fixed_2pct': 'Fixed 2%',
        'fixed_5pct': 'Fixed 5%',
        'atr_2': 'ATR 2x',
        'atr_3': 'ATR 3x',
        'trailing_2pct': 'Trailing 2%',
        'mae_regression': 'MAE Regression'
    }
    colors = {
        'no_stop': 'black',
        'our_model': 'red',
        'our_full': 'darkred',
        'fixed_2pct': 'blue',
        'fixed_5pct': 'orange',
        'atr_2': 'green',
        'atr_3': 'purple',
        'trailing_2pct': 'brown',
        'mae_regression': 'grey'
    }

    # 收集所有股票的原始权益曲线（不做任何填充/截断）
    all_curves = {m: [] for m in methods}
    for code, eq_list in all_equity_data.items():
        for record in eq_list:
            if record['cost'] != cost_bps:
                continue
            eq_no_stop = record['no_stop']
            eq_our = record['our']
            eq_base = record['baselines']

            # 准备各方法的曲线（必须转换为 Python 列表）
            curves = {
                'no_stop': list(eq_no_stop),
                'our_model': list(eq_our),
            }
            curves.update({k: list(v) for k, v in eq_base.items()})

            for m in methods:
                if m in curves and len(curves[m]) >= 2:  # 至少两个点才能有意义
                    all_curves[m].append(curves[m])
            break  # 每只股票每个 cost 只取一个记录

    # 将所有曲线插值到统一的采样点数（target_points）
    target_points = 200
    avg_curves = {}
    for m in methods:
        if not all_curves[m]:
            continue
        interpolated = []
        for curve in all_curves[m]:
            if len(curve) < 2:
                continue
            orig_idx = np.linspace(0, 1, len(curve))
            target_idx = np.linspace(0, 1, target_points)
            interp = np.interp(target_idx, orig_idx, curve)
            interpolated.append(interp)

        if not interpolated:
            continue
        # 转换为二维数组 (n_stocks, target_points) 并计算平均
        arr = np.array(interpolated, dtype=np.float64)  # 此时形状一定均匀
        avg_curve = arr.mean(axis=0)
        avg_curves[m] = avg_curve

    # 绘制平均累积收益曲线
    plt.figure(figsize=(14, 7))
    for m in methods:
        if m in avg_curves:
            # 转换为相对于起点的累计收益率（起点为0）
            cum_return = avg_curves[m] / avg_curves[m][0] - 1
            plt.plot(cum_return, label=method_labels.get(m, m),
                     color=colors.get(m, None), linewidth=2)

    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title(f'Average Cumulative Return (cost={cost_bps}bps, seed={seed})', fontsize=14)
    plt.xlabel('Normalized Trading Step')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(cfg.plot_dir, f'avg_equity_cost{cost_bps}bps_seed{seed}.png')
    plt.savefig(out_path, dpi=cfg.plot_dpi)
    plt.close()
    print(f"Average equity curve saved to {out_path}")


def plot_risk_metrics(summary_df, cfg):
    """
    绘制各方法的夏普比率、最大回撤、期望损失对比柱状图。
    summary_df: 包含 method, sharpe, max_dd, expectile_loss 等列
    """
    os.makedirs(cfg.plot_dir, exist_ok=True)

    # 只取 cost=0 的数据为例
    df = summary_df[summary_df['cost_bps'] == 0].copy()
    if df.empty:
        print("No data for cost=0 to plot metrics.")
        return

    # 分组统计均值与标准误
    metrics = ['sharpe', 'max_dd', 'expectile_loss']
    metric_labels = {'sharpe': 'Sharpe Ratio', 'max_dd': 'Max Drawdown',
                     'expectile_loss': 'Expected Shortfall (5%)'}
    methods_order = ['no_stop', 'our_model', 'our_full', 'fixed_2pct', 'fixed_5pct',
                     'atr_2', 'atr_3', 'trailing_2pct', 'mae_regression']
    colors = ['black', 'red', 'darkred', 'blue', 'orange', 'green', 'purple', 'brown', 'grey']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric in zip(axes, metrics):
        means = []
        errors = []
        labels = []
        for m in methods_order:
            sub = df[df['method'] == m][metric]
            if sub.empty:
                continue
            means.append(sub.mean())
            errors.append(sub.std() / np.sqrt(len(sub)))
            labels.append(m)

        ax.bar(labels, means, yerr=errors, capsize=5, color=colors[:len(labels)])
        ax.set_title(metric_labels[metric])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Risk & Performance Comparison (cost=0bps)', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(cfg.plot_dir, f'risk_metrics_seed{cfg.single_seed}.png')
    plt.savefig(out_path, dpi=cfg.plot_dpi)
    plt.close()
    print(f"Risk metrics plot saved to {out_path}")


def main():
    if cfg.run_multi_seed:
        all_summaries = []
        for seed in cfg.seeds:
            print(f"\n=== Running seed {seed} ===")
            summ = run_one_seed(seed)
            if summ is not None:
                summ['seed'] = seed
                all_summaries.append(summ)
        if all_summaries:
            full_df = pd.concat(all_summaries)
            full_df.to_csv(os.path.join(cfg.results_dir, "summary_all_seeds.csv"), index=False)
    else:
        run_one_seed(cfg.single_seed)


if __name__ == "__main__":
    main()
