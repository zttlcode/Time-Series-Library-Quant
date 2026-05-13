"""
对比尾部感知元标签 (CTAML)：完整实验代码
根据论文"Contrastive Tail-Aware Meta-Labeling: Self-Supervised Extreme Region Distance Field Learning for Trading Signal Filtering"
目标期刊：ESWA / EAAI / ASOC / Neurocomputing / KBS / Information Sciences

依赖：pandas, numpy, torch, scipy, sklearn, matplotlib, tqdm
"""
# 导入必要的科学计算、数据处理和深度学习库
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.stats import genpareto  # 用于广义帕累托分布拟合
from scipy.signal import find_peaks  # 用于极值点检测（备用）
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import random
import warnings

warnings.filterwarnings('ignore')  # 忽略警告，保持输出整洁


# ================== 配置 ==================
class Config:
    # 数据
    signal_file = ""
    ohlcv_file = ""

    # 窗口与距离参数
    window_size = 160
    future_T = 60
    kappa = 1.5

    # EVT相关
    evt_rolling = 252
    evt_quantile = 0.90
    extrema_win = 3

    # 对比学习
    contrastive_win = 5
    D_buckets = 8

    # 标签与过滤
    D_threshold = 0.4
    oracle_D_threshold = 0.4
    good_D_threshold = 0.4

    # 当前只训练 buy 过滤器
    train_signal_type = 'buy'

    # 使用分类概率过滤
    use_good_classifier = True

    # 目标交易覆盖率。注意：现在只用于全局验证集阈值的备选方案
    desired_trade_ratio = 0.30

    # 阈值校准方式：
    # 'global_f1'：用所有验证集 buy 信号，根据真实 good_label 搜索最佳 F1 阈值
    # 'global_quantile'：用所有验证集 buy 信号 p_good 的分位数阈值
    threshold_method = 'global_f1_constrained'

    # 如果验证集无法计算阈值，则使用该阈值
    fallback_pgood_threshold = 0.5

    # 模型
    encoder_type = 'BiLSTM'
    d_model = 64
    n_heads = 8
    proj_dim = 64
    quantiles = [0.1, 0.5, 0.9]
    use_taa = False
    lam_init = 0.1
    tau_cl = 0.1

    # 当前阶段：分类为主，先关闭 D 和 xi 的辅助损失，避免小样本下互相干扰
    lambda_cls = 1.0
    lambda_D = 0.0
    lambda_xi = 0.0
    lambda_cl = 0.0

    # 训练
    batch_size = 32
    lr = 5e-4
    epochs = 120
    patience = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 均衡采样和早停
    use_balanced_sampler = True
    early_stop_metric = 'val_auc'  # 'val_auc' or 'val_ap' or 'val_loss'
    seed = 42

    # 回测
    stop_loss_c = 1.0
    stop_loss_alpha = 0.05


    """ 四组实验
    strategy_name = 'BOLL_RSI'
    market_name = 'A'
    run_tag = 'BOLL_RSI_A'
    
    strategy_name = 'BOLL_RSI'
    market_name = 'US'
    run_tag = 'BOLL_RSI_US'
    
    strategy_name = 'MACD_KDJ'
    market_name = 'A'
    run_tag = 'MACD_KDJ_A'
    
    strategy_name = 'MACD_KDJ'
    market_name = 'US'
    run_tag = 'MACD_KDJ_US'
    """
    # 实验管理
    strategy_name = 'BOLL_RSI'   # 可选 'BOLL_RSI' / 'MACD_KDJ'
    market_name = 'A'            # 可选 'A' / 'US'
    results_dir = './ctaml_results'
    run_tag = 'BOLL_RSI_A'       # 可选 'BOLL_RSI' / 'MACD_KDJ'

    # 阈值覆盖率约束，避免 global_f1 退化为近似全买
    min_val_coverage = 0.20
    max_val_coverage = 0.45

    # 是否保存结果
    save_outputs = True
    # 是否保存每只股票的逐日收益率曲线
    save_equity_curves = True

cfg = Config()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ================== 1. 数据加载与特征工程 ==================
def load_and_prepare(signal_path, ohlcv_path):
    """
    读取原始OHLCV数据与交易信号，计算技术指标，生成特征矩阵。
    返回特征DataFrame和特征列名列表。
    """
    # 读取交易信号文件，解析时间列
    signals = pd.read_csv(signal_path, parse_dates=['time'])
    # 读取OHLCV文件，解析时间列
    ohlcv = pd.read_csv(ohlcv_path, parse_dates=['time'])
    # 以时间为索引，按时间排序确保序列顺序正确
    df = ohlcv.set_index('time').sort_index()

    # 计算对数收益率及其绝对值，后者用于尾部注意力模块
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_returns'] = df['returns'].abs()

    # 计算不同周期的移动平均（价格均线和成交量均线）
    for w in [5, 10, 20, 60]:
        df[f'ma_{w}'] = df['close'].rolling(w).mean()
        df[f'vol_ma_{w}'] = df['volume'].rolling(w).mean()

    # MACD：指数加权移动平均线的差值，以及信号线和柱
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (14日)：衡量超买超卖的经典指标
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # KDJ (简化版)：基于14日最高/最低价计算RSV，再平滑得到K、D、J
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    rsv = (df['close'] - low_14) / (high_14 - low_14) * 100
    df['k'] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df['d'] = df['k'].ewm(alpha=1 / 3, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    # ATR (14)：真实波幅均值，用于距离归一化和波动率衡量
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 价格和成交量的单步变化率，捕捉短期动量
    df['pct_chg'] = df['close'].pct_change()
    df['vol_chg'] = df['volume'].pct_change()

    # 将无穷值替换为 NaN，避免后续 StandardScaler 报错
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 删除含有NaN的行（由rolling计算产生）
    df.dropna(inplace=True)

    # 选定的特征列清单，包含原始价量、技术指标和衍生特征
    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'returns', 'abs_returns',
                    'ma_5', 'ma_10', 'ma_20', 'ma_60',
                    'vol_ma_5', 'vol_ma_10', 'vol_ma_20', 'vol_ma_60',
                    'macd', 'macd_signal', 'macd_hist',
                    'rsi', 'k', 'd', 'j', 'atr', 'pct_chg', 'vol_chg']
    df = df[feature_cols]  # 仅保留特征列

    # 将信号数据并入主DataFrame（左连接），未出现信号的时间点填充'none'
    signals = signals.set_index('time').sort_index()
    df = df.join(signals[['price', 'signal']], how='left')
    df['signal'] = df['signal'].fillna('none')
    return df, feature_cols


# ================== 2. 标注函数 ==================
def target_extrema_type(signal_type):
    """
    对于 long-only 震荡/均值回归策略：
    buy  应该靠近 valley
    sell 应该靠近 peak
    """
    if signal_type == 'buy':
        return 'valley'
    elif signal_type == 'sell':
        return 'peak'
    else:
        return None


def print_label_diagnostics(code, df, signal_indices=None, prefix=""):
    """
    打印 D_label 分布诊断。
    """
    if signal_indices is None:
        sig_df = df[df['signal'] != 'none'].copy()
    else:
        sig_df = df.iloc[signal_indices]
        sig_df = sig_df[sig_df['signal'] != 'none'].copy()

    if len(sig_df) == 0:
        print(f"{prefix}股票 {code}: 无信号可诊断")
        return

    valid = sig_df.dropna(subset=['D_label'])
    if len(valid) == 0:
        print(f"{prefix}股票 {code}: 信号 D_label 全为空")
        return

    d = valid['D_label']
    d1_ratio = (d >= 0.999999).mean()
    non_d1 = (d < 0.999999).sum()

    print(f"\n===== {prefix}股票 {code} 标签诊断 =====")
    print(f"信号数: {len(valid)}")
    print(f"D_label min/median/mean/max: {d.min():.4f} / {d.median():.4f} / {d.mean():.4f} / {d.max():.4f}")
    print(f"D=1 占比: {d1_ratio:.2%}, 非 D=1 数量: {non_d1}")
    print(valid.groupby('signal')['D_label'].agg(['count', 'mean', 'median', 'min', 'max']).to_string())


def compute_evt_distance_xi(df, cfg):
    """
    核心标注函数：
    1. 基于滚动窗口拟合 GPD，产生时变形状参数 ξ；
    2. 识别结构性极值点，并保存极值方向：peak / valley；
    3. 对交易信号计算 ERDF 距离标签 D。

    重要修正：
    - buy 匹配 valley
    - sell 匹配 peak
    - 搜索从 idx_s 当天开始，允许信号当天就是极值点
    - D 裁剪到 [0, 1]
    - xi_series 做 ffill/bfill，避免大量信号因 xi NaN 被丢弃
    """
    returns = df['returns'].values
    abs_r = df['abs_returns'].values
    n = len(df)

    xi_series = np.full(n, np.nan)
    sigma_series = np.full(n, np.nan)

    threshold_series = pd.Series(abs_r).rolling(
        cfg.evt_rolling,
        min_periods=cfg.evt_rolling
    ).quantile(cfg.evt_quantile).values

    # ---------- 1. 滚动 GPD 拟合 ----------
    for i in range(cfg.evt_rolling, n):
        window_abs = abs_r[i - cfg.evt_rolling:i]
        thresh = threshold_series[i]

        if np.isnan(thresh):
            continue

        exceed = window_abs[window_abs > thresh] - thresh

        if len(exceed) >= 10:
            try:
                shape, _, scale = genpareto.fit(exceed)
                xi_series[i] = shape
                sigma_series[i] = scale
            except Exception:
                pass

    # 防止 xi_label 大量 NaN 导致样本被过滤
    xi_series = pd.Series(xi_series).ffill().bfill().fillna(0.0).values
    sigma_series = pd.Series(sigma_series).ffill().bfill().fillna(np.nanstd(abs_r) + 1e-6).values

    # ---------- 2. EVT 极端事件 ----------
    extreme_mask = (abs_r > threshold_series) & (~np.isnan(threshold_series))

    # extrema_info: idx -> 'peak' or 'valley'
    extrema_info = {}

    for i in np.where(extreme_mask)[0]:
        local_start = max(0, i - cfg.extrema_win)
        local_end = min(n - 1, i + cfg.extrema_win)
        local_close = df['close'].iloc[local_start:local_end + 1]

        if len(local_close) == 0:
            continue

        if returns[i] > 0:
            # 正极端收益：结构性高点 peak
            t_max = local_close.idxmax()
            idx_ext = df.index.get_loc(t_max)
            extrema_info[idx_ext] = 'peak'
        else:
            # 负极端收益：结构性低点 valley
            t_min = local_close.idxmin()
            idx_ext = df.index.get_loc(t_min)
            extrema_info[idx_ext] = 'valley'

    # ---------- 3. 对信号计算 D_label ----------
    D_labels = []
    xi_labels = []

    signal_times = df[df['signal'] != 'none'].index

    for t_s in signal_times:
        idx_s = df.index.get_loc(t_s)
        signal_type = df['signal'].loc[t_s]
        target_type = target_extrema_type(signal_type)

        future_end = min(n, idx_s + cfg.future_T + 1)
        min_dist = 1.0

        if target_type is not None:
            # 注意：从 idx_s 开始，而不是 idx_s + 1
            # 因为 BOLL+RSI 的 buy/sell 很可能当天就是 valley/peak
            for idx_t in range(idx_s, future_end):
                if idx_t not in extrema_info:
                    continue

                ext_type = extrema_info[idx_t]

                if ext_type != target_type:
                    continue

                P_ext = df['close'].iloc[idx_t]
                P_s = df['close'].iloc[idx_s]
                atr_s = df['atr'].iloc[idx_s]

                if np.isnan(atr_s) or atr_s <= 1e-12:
                    continue

                time_diff = idx_t - idx_s
                price_diff = abs(P_s - P_ext)

                dist = np.sqrt(
                    (time_diff / cfg.future_T) ** 2 +
                    (price_diff / (cfg.kappa * atr_s)) ** 2
                )

                min_dist = min(1.0, dist)
                break

        D_labels.append(min_dist)
        xi_labels.append(xi_series[idx_s])

    df_out = df.copy()
    df_out['D_label'] = np.nan
    df_out['xi_label'] = np.nan

    locs = [df.index.get_loc(t) for t in signal_times]
    df_out.iloc[locs, df_out.columns.get_loc('D_label')] = D_labels
    df_out.iloc[locs, df_out.columns.get_loc('xi_label')] = xi_labels

    return df_out, xi_series, sigma_series, extrema_info


# ================== 3. 数据集与对比学习准备 ==================
class SignalDataset(Dataset):
    """
    买入信号过滤数据集：
    - x_win
    - D_label
    - xi_label
    - good_label: D_label <= cfg.good_D_threshold
    - abs_ret
    - pos_win
    - neg_win
    """

    def __init__(self, df, feature_cols, cfg, extrema_info, scaler=None, signal_indices=None, signal_filter_type=None):
        self.df = df
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels_D = df['D_label'].values
        self.labels_xi = df['xi_label'].values
        self.window = cfg.window_size
        self.cfg = cfg
        self.signal_filter_type = signal_filter_type

        if isinstance(extrema_info, dict):
            self.extrema_info = extrema_info
        else:
            self.extrema_info = {int(e): 'unknown' for e in extrema_info}

        self.extrema_indices = set(self.extrema_info.keys())

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

        if signal_indices is not None:
            raw_indices = signal_indices
        else:
            raw_indices = np.where((df['signal'] != 'none').values)[0]

        self.signal_indices = []
        for i in raw_indices:
            if i < self.window:
                continue
            if i + cfg.future_T + cfg.contrastive_win >= len(df):
                continue
            if np.isnan(self.labels_D[i]) or np.isnan(self.labels_xi[i]):
                continue

            sig = self.df['signal'].iloc[i]

            # 关键：只训练 buy 过滤器
            if self.signal_filter_type is not None and sig != self.signal_filter_type:
                continue

            self.signal_indices.append(i)

        self.signal_types = [self.df['signal'].iloc[i] for i in self.signal_indices]
        self.good_labels = np.array([
            1.0 if self.labels_D[i] <= self.cfg.good_D_threshold else 0.0
            for i in self.signal_indices
        ], dtype=np.float32)

    def get_good_labels(self):
        return self.good_labels

    def __len__(self):
        return len(self.signal_indices)

    def __getitem__(self, idx):
        i = self.signal_indices[idx]

        x_win = self.features[i - self.window:i]
        d_label = float(self.labels_D[i])
        xi_label = float(self.labels_xi[i])
        good_label = 1.0 if d_label <= self.cfg.good_D_threshold else 0.0

        abs_ret = self.df['abs_returns'].values[i - self.window:i]

        pos_len = 2 * self.cfg.contrastive_win + 1
        pos_win = np.zeros((pos_len, self.features.shape[1]), dtype=np.float32)

        future_end = min(len(self.df), i + self.cfg.future_T + 1)
        sig_type = self.signal_types[idx]
        target_type = target_extrema_type(sig_type)

        if target_type is not None:
            for j in range(i, future_end):
                if j in self.extrema_info and self.extrema_info[j] == target_type:
                    start = j - self.cfg.contrastive_win
                    end = j + self.cfg.contrastive_win + 1
                    if start >= 0 and end <= len(self.df):
                        pos_win = self.features[start:end]
                    break

        neg_win = np.zeros_like(pos_win)
        candidates = []

        max_start = len(self.df) - pos_len
        if max_start > self.window:
            for _ in range(20):
                k = np.random.randint(self.window, max_start)
                if all(abs(k - e) > self.cfg.future_T / 2 for e in self.extrema_indices):
                    candidates.append(k)

        if candidates:
            k = np.random.choice(candidates)
            neg_win = self.features[k:k + pos_len]

        return torch.tensor(x_win, dtype=torch.float32), \
            torch.tensor(d_label, dtype=torch.float32), \
            torch.tensor(xi_label, dtype=torch.float32), \
            torch.tensor(good_label, dtype=torch.float32), \
            torch.tensor(abs_ret, dtype=torch.float32), \
            torch.tensor(pos_win, dtype=torch.float32), \
            torch.tensor(neg_win, dtype=torch.float32)


def get_concat_good_labels(concat_dataset):
    """
    从 ConcatDataset 中提取所有子数据集的 good_labels。
    """
    labels = []

    for ds in concat_dataset.datasets:
        if hasattr(ds, "get_good_labels"):
            labels.extend(ds.get_good_labels().tolist())
        else:
            raise ValueError("子数据集缺少 get_good_labels() 方法")

    return np.array(labels, dtype=np.float32)


def build_balanced_sampler(concat_dataset):
    """
    对 good/bad 样本做均衡采样，缓解 good 信号稀缺和输出塌缩。
    """
    labels = get_concat_good_labels(concat_dataset)

    if len(labels) == 0:
        return None

    pos = labels.sum()
    neg = len(labels) - pos

    if pos < 1 or neg < 1:
        return None

    weight_pos = 0.5 / pos
    weight_neg = 0.5 / neg

    weights = np.where(labels > 0.5, weight_pos, weight_neg)
    weights = torch.tensor(weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    print("\n===== 训练采样诊断 =====")
    print(f"训练 buy 样本数: {len(labels)}")
    print(f"good 样本数: {int(pos)}, bad 样本数: {int(neg)}")
    print(f"good 占比: {pos / len(labels):.2%}")
    print("已启用 WeightedRandomSampler 做 good/bad 均衡采样")

    return sampler


# ================== 4. 模型定义 ==================
class TailAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads, lam_init=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.lam = nn.Parameter(torch.tensor(lam_init))

    def forward(self, x, abs_returns, xi_hat, sigma_hat):
        # x: (B, T, d_model)
        if xi_hat.dim() == 1:
            xi_hat = xi_hat.unsqueeze(-1).expand(-1, x.size(1))  # xi_hat: (B,) 或 (B, T)
            sigma_hat = sigma_hat.unsqueeze(-1).expand(-1, x.size(1))  # sigma_hat: (B,) 或 (B, T)

        # 对 xi_hat 做保护：将绝对值小于 1e-6 的值替换为一个很小的正数，避免除零
        xi_hat = torch.where(
            torch.abs(xi_hat) < 1e-6,
            torch.tensor(1e-6, device=xi_hat.device, dtype=xi_hat.dtype),
            xi_hat
        )

        z = abs_returns / (sigma_hat + 1e-8)  # abs_returns: (B, T)

        xi_z = xi_hat * z
        # 保证 1 + xi_z > 0，即 xi_z > -1
        xi_z = torch.clamp(xi_z, min=-0.999, max=1e6)  # 上界也限制一下，防止过大
        log_surv = -1.0 / (xi_hat + 1e-6) * torch.log1p(xi_z + 1e-8)

        bias = self.lam * log_surv.unsqueeze(1)  # (B, 1, T)
        # 扩展为 (B, T, T) 再适配多头
        bias = bias.expand(-1, x.size(1), -1)  # (B, T, T)
        attn_out, _ = self.attn(x, x, x, attn_mask=bias.repeat(self.attn.num_heads, 1, 1))
        return attn_out


class QuantileHead(nn.Module):
    def __init__(self, in_dim, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        self.net = nn.Linear(in_dim, len(quantiles))

    def forward(self, h):
        return torch.sigmoid(self.net(h))


class ProjHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


# BiLSTM 编码器（输出序列）
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, d_model // 2, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, d_model)
        return self.proj(out)  # 保持序列


# Informer 编码器（输出序列）
class InformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, seq_len, n_heads=8, e_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        from layers.Embed import DataEmbedding
        from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
        from layers.SelfAttention_Family import ProbAttention, AttentionLayer

        self.embedding = DataEmbedding(input_dim, d_model, 'fixed', 'd', dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, 3, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu'
                ) for _ in range(e_layers)
            ],
            None,  # 不使用 ConvLayer，避免序列长度变化 [ConvLayer(d_model) for _ in range(e_layers-1)] if e_layers>1 else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        enc_out = self.embedding(x, None)  # (B, T, d_model)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.proj(enc_out)


class CTAML(nn.Module):
    def __init__(self, encoder, d_model, use_taa=True, proj_dim=128,
                 n_heads=8, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.encoder = encoder
        self.use_taa = use_taa
        self.taa = TailAwareAttention(d_model, n_heads) if use_taa else None

        self.head_D = QuantileHead(d_model, quantiles)
        self.head_xi = nn.Linear(d_model, 1)
        self.proj_head = ProjHead(d_model, proj_dim)

        # 新增：好买点分类头
        self.head_good = nn.Linear(d_model, 1)

    def forward(self, x, abs_returns, xi_hat, sigma_hat):
        h_seq = self.encoder(x)

        if self.use_taa and abs_returns is not None:
            h_seq = self.taa(h_seq, abs_returns, xi_hat, sigma_hat)

        h = h_seq[:, -1, :]

        D_hat = self.head_D(h)
        xi_out = self.head_xi(h).squeeze(-1)
        proj = self.proj_head(h)
        good_logit = self.head_good(h).squeeze(-1)

        return D_hat, xi_out, proj, good_logit


# ================== 5. 损失函数 ==================
def pinball_loss(y_hat, y, quantiles):
    # y_hat: (B, n_q)   y: (B,)
    losses = []
    for i, q in enumerate(quantiles):
        e = y - y_hat[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.stack(losses, -1).mean()


def info_nce(anchors, positives, tau=0.1):
    # anchors, positives: (B, proj_dim)
    logits = anchors @ positives.t() / tau
    labels = torch.arange(len(anchors), device=logits.device)
    return F.cross_entropy(logits, labels)


def weighted_bce_loss(logits, targets):
    """
    BCE 分类损失。
    当前训练已使用 WeightedRandomSampler 均衡 good/bad，
    因此这里不再使用过强的动态 pos_weight，避免概率输出被压缩。
    """
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


# ================== 6. 训练循环 ==================
def train_epoch(model, dataloader, optimizer, cfg):
    model.train()
    total_loss = 0.0

    for x, d_label, xi_label, good_label, abs_ret, pos_win, _ in dataloader:
        x = x.to(cfg.device)
        d_label = d_label.to(cfg.device)
        xi_label = xi_label.to(cfg.device)
        good_label = good_label.to(cfg.device)
        abs_ret = abs_ret.to(cfg.device)
        pos_win = pos_win.to(cfg.device)

        sigma_hat = torch.std(abs_ret, dim=1) + 1e-6

        D_hat, xi_out, proj_anchor, good_logit = model(
            x, abs_ret, xi_label, sigma_hat
        )

        loss_cls = weighted_bce_loss(good_logit, good_label)
        loss_D = pinball_loss(D_hat, d_label, cfg.quantiles)
        loss_xi = F.smooth_l1_loss(xi_out, xi_label)

        # 对比学习先支持，但默认 lambda_cl=0
        if cfg.lambda_cl > 0:
            h_pos_seq = model.encoder(pos_win)
            h_pos = h_pos_seq[:, -1, :]
            proj_pos = model.proj_head(h_pos)

            pos_valid_mask = pos_win.reshape(pos_win.size(0), -1).abs().sum(dim=1) > 1e-8
            if pos_valid_mask.sum() > 1:
                loss_cl = info_nce(
                    proj_anchor[pos_valid_mask],
                    proj_pos[pos_valid_mask],
                    cfg.tau_cl
                )
            else:
                loss_cl = torch.tensor(0.0, device=cfg.device)
        else:
            loss_cl = torch.tensor(0.0, device=cfg.device)

        loss = (
                cfg.lambda_cls * loss_cls +
                cfg.lambda_D * loss_D +
                cfg.lambda_xi * loss_xi +
                cfg.lambda_cl * loss_cl
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def eval_epoch(model, dataloader, cfg):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, d_label, xi_label, good_label, abs_ret, pos_win, _ in dataloader:
            x = x.to(cfg.device)
            d_label = d_label.to(cfg.device)
            xi_label = xi_label.to(cfg.device)
            good_label = good_label.to(cfg.device)
            abs_ret = abs_ret.to(cfg.device)

            sigma_hat = torch.std(abs_ret, dim=1) + 1e-6

            D_hat, xi_out, _, good_logit = model(
                x, abs_ret, xi_label, sigma_hat
            )

            loss_cls = weighted_bce_loss(good_logit, good_label)
            loss_D = pinball_loss(D_hat, d_label, cfg.quantiles)
            loss_xi = F.smooth_l1_loss(xi_out, xi_label)

            loss = (
                    cfg.lambda_cls * loss_cls +
                    cfg.lambda_D * loss_D +
                    cfg.lambda_xi * loss_xi
            )

            total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def eval_classifier_metrics(model, dataloader, cfg):
    """
    在验证集上评估分类排序质量。
    返回 loss、AUC、AP、p_good 分布等。
    """
    model.eval()

    all_y = []
    all_p = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, d_label, xi_label, good_label, abs_ret, pos_win, _ in dataloader:
            x = x.to(cfg.device)
            d_label = d_label.to(cfg.device)
            xi_label = xi_label.to(cfg.device)
            good_label = good_label.to(cfg.device)
            abs_ret = abs_ret.to(cfg.device)

            sigma_hat = torch.std(abs_ret, dim=1) + 1e-6

            D_hat, xi_out, _, good_logit = model(
                x, abs_ret, xi_label, sigma_hat
            )

            loss_cls = weighted_bce_loss(good_logit, good_label)

            loss = cfg.lambda_cls * loss_cls

            if cfg.lambda_D > 0:
                loss_D = pinball_loss(D_hat, d_label, cfg.quantiles)
                loss = loss + cfg.lambda_D * loss_D

            if cfg.lambda_xi > 0:
                loss_xi = F.smooth_l1_loss(xi_out, xi_label)
                loss = loss + cfg.lambda_xi * loss_xi

            total_loss += loss.item()
            n_batches += 1

            p_good = torch.sigmoid(good_logit).detach().cpu().numpy()
            y = good_label.detach().cpu().numpy()

            all_p.extend(p_good.tolist())
            all_y.extend(y.tolist())

    all_p = np.array(all_p, dtype=float)
    all_y = np.array(all_y, dtype=float)

    avg_loss = total_loss / max(1, n_batches)

    if len(np.unique(all_y)) < 2:
        auc = np.nan
        ap = np.nan
    else:
        auc = roc_auc_score(all_y, all_p)
        ap = average_precision_score(all_y, all_p)

    good_mean = all_p[all_y > 0.5].mean() if np.any(all_y > 0.5) else np.nan
    bad_mean = all_p[all_y <= 0.5].mean() if np.any(all_y <= 0.5) else np.nan

    return {
        'loss': avg_loss,
        'auc': auc,
        'ap': ap,
        'p_min': float(np.min(all_p)) if len(all_p) else np.nan,
        'p_median': float(np.median(all_p)) if len(all_p) else np.nan,
        'p_max': float(np.max(all_p)) if len(all_p) else np.nan,
        'good_mean': float(good_mean) if np.isfinite(good_mean) else np.nan,
        'bad_mean': float(bad_mean) if np.isfinite(bad_mean) else np.nan,
        'n': len(all_y),
        'good_count': int(np.sum(all_y > 0.5))
    }


def split_signals_train_val_test(df, cfg, train_ratio=0.6, val_ratio=0.2):
    """
    按时间切分信号：
    train / val / test

    train: 训练模型
    val:   早停和阈值校准
    test:  最终评估
    """
    signal_df = df[df['signal'] != 'none'].copy()

    if signal_df.empty:
        return [], [], []

    signal_df = signal_df.sort_index()
    n = len(signal_df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_times = signal_df.index[:train_end]
    val_times = signal_df.index[train_end:val_end]
    test_times = signal_df.index[val_end:]

    train_locs = [df.index.get_loc(t) for t in train_times]
    val_locs = [df.index.get_loc(t) for t in val_times]
    test_locs = [df.index.get_loc(t) for t in test_times]

    # 防止训练样本未来窗口触碰验证集
    if val_locs:
        min_val_loc = min(val_locs)
        train_locs = [
            loc for loc in train_locs
            if loc + cfg.future_T + cfg.contrastive_win < min_val_loc
        ]

    # 防止验证样本未来窗口触碰测试集
    if test_locs:
        min_test_loc = min(test_locs)
        val_locs = [
            loc for loc in val_locs
            if loc + cfg.future_T + cfg.contrastive_win < min_test_loc
        ]

    return train_locs, val_locs, test_locs


# ================== 7. 回测与决策 ==================
def apply_filter_and_trade(df, model, cfg, scaler, feature_cols,
                           signal_locs=None,
                           start_loc=None,
                           end_loc=None,
                           threshold=None):
    """
    使用 p_good 过滤 buy。
    buy: p_good >= threshold 且空仓
    sell: 直接平仓
    """
    if threshold is None:
        threshold = 0.5

    if start_loc is None:
        start_loc = cfg.window_size

    if end_loc is None:
        end_loc = len(df)

    start_loc = max(start_loc, cfg.window_size)
    end_loc = min(end_loc, len(df))

    if signal_locs is None:
        signal_locs = set(np.where((df['signal'] != 'none').values)[0])
    else:
        signal_locs = set(signal_locs)

    capital = 1.0
    position = 0.0
    equity = [capital]
    signal_records = []

    model.eval()

    for i_loc in range(start_loc, end_loc):
        idx = df.index[i_loc]
        row = df.iloc[i_loc]

        if i_loc in signal_locs and row['signal'] != 'none':
            x_win = scaler.transform(
                df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values
            )
            x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)

            abs_ret = torch.tensor(
                df['abs_returns'].iloc[i_loc - cfg.window_size:i_loc].values,
                dtype=torch.float32
            ).unsqueeze(0).to(cfg.device)

            with torch.no_grad():
                D_hat, _, _, good_logit = model(
                    x_ten,
                    abs_ret,
                    torch.zeros(1).to(cfg.device),
                    torch.tensor([1.0]).to(cfg.device)
                )

                d_mid = D_hat[0, 1].item()
                p_good = torch.sigmoid(good_logit)[0].item()

            if row['signal'] == 'buy':
                executed = (p_good >= threshold)

                signal_records.append({
                    'time': idx,
                    'price': row['close'],
                    'd_mid': d_mid,
                    'p_good': p_good,
                    'threshold': threshold,
                    'executed': executed
                })

                if executed and position == 0:
                    position = capital / row['close']
                    capital = 0.0

            elif row['signal'] == 'sell':
                if position > 0:
                    capital = position * row['close']
                    position = 0.0

        if position > 0:
            equity.append(position * row['close'])
        else:
            equity.append(capital)

    return equity, signal_records


def backtest_no_filter(df, cfg, signal_locs=None, start_loc=None, end_loc=None):
    """
    无过滤基线。
    重要修正：
    - 只在指定测试集 signal_locs 上执行交易；
    - 从 start_loc 开始，避免全样本混测。
    """
    if start_loc is None:
        start_loc = 0

    if end_loc is None:
        end_loc = len(df)

    start_loc = max(0, start_loc)
    end_loc = min(end_loc, len(df))

    if signal_locs is None:
        signal_locs = set(np.where((df['signal'] != 'none').values)[0])
    else:
        signal_locs = set(signal_locs)

    capital = 1.0
    position = 0.0
    equity = [capital]

    for i_loc in range(start_loc, end_loc):
        row = df.iloc[i_loc]

        if i_loc in signal_locs and row['signal'] != 'none':
            if row['signal'] == 'buy' and position == 0:
                position = capital / row['close']
                capital = 0.0
            elif row['signal'] == 'sell' and position > 0:
                capital = position * row['close']
                position = 0.0

        if position > 0:
            equity.append(position * row['close'])
        else:
            equity.append(capital)

    return equity


def backtest_oracle_label(df, cfg, signal_locs=None, start_loc=None, end_loc=None, threshold=None):
    """
    Oracle 标签回测：
    直接使用真实 D_label 过滤 buy 信号。

    注意：
    - 这是诊断用，不是可实盘结果；
    - 如果 Oracle 都无法优于无过滤，说明 D_label 设计和策略收益目标不匹配。
    """
    if threshold is None:
        threshold = cfg.oracle_D_threshold

    if start_loc is None:
        start_loc = 0

    if end_loc is None:
        end_loc = len(df)

    start_loc = max(0, start_loc)
    end_loc = min(end_loc, len(df))

    if signal_locs is None:
        signal_locs = set(np.where((df['signal'] != 'none').values)[0])
    else:
        signal_locs = set(signal_locs)

    capital = 1.0
    position = 0.0
    equity = [capital]
    records = []

    for i_loc in range(start_loc, end_loc):
        idx = df.index[i_loc]
        row = df.iloc[i_loc]

        if i_loc in signal_locs and row['signal'] != 'none':
            if row['signal'] == 'buy':
                d_label = row.get('D_label', np.nan)
                executed = (not np.isnan(d_label)) and (d_label <= threshold)

                records.append({
                    'time': idx,
                    'price': row['close'],
                    'D_label': d_label,
                    'threshold': threshold,
                    'executed': executed
                })

                if executed and position == 0:
                    position = capital / row['close']
                    capital = 0.0

            elif row['signal'] == 'sell' and position > 0:
                capital = position * row['close']
                position = 0.0

        if position > 0:
            equity.append(position * row['close'])
        else:
            equity.append(capital)

    return equity, records


def predict_for_indices(df, model, cfg, scaler, feature_cols, signal_indices):
    """
    对指定信号位置预测：
    - d_mid
    - p_good
    """
    records = []
    model.eval()

    for i_loc in signal_indices:
        if i_loc < cfg.window_size:
            continue

        row = df.iloc[i_loc]

        if row['signal'] == 'none':
            continue

        x_win = scaler.transform(
            df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values
        )
        x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)

        abs_ret = torch.tensor(
            df['abs_returns'].iloc[i_loc - cfg.window_size:i_loc].values,
            dtype=torch.float32
        ).unsqueeze(0).to(cfg.device)

        with torch.no_grad():
            D_hat, _, _, good_logit = model(
                x_ten,
                abs_ret,
                torch.zeros(1).to(cfg.device),
                torch.tensor([1.0]).to(cfg.device)
            )

            d_mid = D_hat[0, 1].item()
            p_good = torch.sigmoid(good_logit)[0].item()

        records.append({
            'loc': i_loc,
            'time': df.index[i_loc],
            'signal': row['signal'],
            'price': row['close'],
            'd_mid': d_mid,
            'p_good': p_good
        })

    return pd.DataFrame(records)


def attach_labels_to_prediction(df, pred_df, good_threshold=0.4):
    """
    给预测结果附加 D_label 和 true_good。
    """
    if pred_df is None or len(pred_df) == 0:
        return pd.DataFrame()

    label_df = df[['D_label']].copy()
    label_df = label_df.reset_index().rename(columns={'index': 'time'})
    label_df['time'] = pd.to_datetime(label_df['time'])

    out = pred_df.copy()
    out['time'] = pd.to_datetime(out['time'])
    out = out.merge(label_df, on='time', how='left')
    out['true_good'] = out['D_label'] <= good_threshold

    return out


def calibrate_global_threshold_from_val(val_pred_all, cfg):
    """
    用所有股票验证期 buy 信号统一校准 p_good 阈值。

    支持:
    - global_quantile: 固定目标覆盖率
    - global_f1: 原始F1
    - global_f1_constrained: 带覆盖率约束的F1，推荐主实验使用
    """
    if val_pred_all is None or len(val_pred_all) == 0:
        print("全局验证集为空，使用 fallback 阈值")
        return cfg.fallback_pgood_threshold

    dfv = val_pred_all[val_pred_all['signal'] == 'buy'].dropna(subset=['p_good', 'D_label']).copy()

    if len(dfv) == 0:
        print("全局验证集 buy 为空，使用 fallback 阈值")
        return cfg.fallback_pgood_threshold

    y = (dfv['D_label'].values <= cfg.good_D_threshold).astype(int)
    p = dfv['p_good'].values.astype(float)

    base_rate = y.mean()

    print("\n===== 全局验证集 p_good 阈值校准 =====")
    print(f"验证 buy 数: {len(dfv)}")
    print(f"验证 good 数: {int(y.sum())}, good 占比: {base_rate:.2%}")
    print(f"p_good min/median/max: {p.min():.4f} / {np.median(p):.4f} / {p.max():.4f}")

    if len(np.unique(y)) < 2:
        q = 1.0 - cfg.desired_trade_ratio
        theta = float(np.quantile(p, q))
        print(f"验证集类别单一，使用分位数阈值: {theta:.4f}")
        return theta

    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    print(f"验证集 AUC={auc:.4f}, AP={ap:.4f}, AP/base={ap / max(base_rate, 1e-12):.4f}")

    # 固定目标覆盖率
    if cfg.threshold_method == 'global_quantile':
        q = 1.0 - cfg.desired_trade_ratio
        theta = float(np.quantile(p, q))
        pred = (p >= theta).astype(int)
        print(
            f"使用 global_quantile，目标执行比例 {cfg.desired_trade_ratio:.2%}, "
            f"threshold={theta:.4f}, 实际覆盖率={pred.mean():.2%}"
        )
        return theta

    # F1 或带覆盖率约束 F1
    candidates = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 193)))

    best_theta = cfg.fallback_pgood_threshold
    best_score = -1.0
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_cov = 0.0

    valid_candidate_count = 0

    for th in candidates:
        pred = (p >= th).astype(int)

        if pred.sum() == 0:
            continue

        cov = pred.mean()

        # 关键：覆盖率约束，防止 F1 阈值退化为过度交易
        if cfg.threshold_method == 'global_f1_constrained':
            if cov < cfg.min_val_coverage or cov > cfg.max_val_coverage:
                continue

        valid_candidate_count += 1

        f1 = f1_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)

        # 在F1基础上轻微鼓励precision，避免高召回低精度
        score = f1 + 0.10 * prec

        if score > best_score:
            best_score = score
            best_theta = float(th)
            best_f1 = float(f1)
            best_prec = float(prec)
            best_rec = float(rec)
            best_cov = float(cov)

    if valid_candidate_count == 0:
        q = 1.0 - cfg.desired_trade_ratio
        best_theta = float(np.quantile(p, q))
        pred = (p >= best_theta).astype(int)
        best_cov = pred.mean()
        best_prec = precision_score(y, pred, zero_division=0)
        best_rec = recall_score(y, pred, zero_division=0)
        best_f1 = f1_score(y, pred, zero_division=0)

        print("无满足覆盖率约束的候选阈值，回退到 global_quantile")

    print(f"使用 {cfg.threshold_method} threshold={best_theta:.4f}")
    print(
        f"验证集 precision={best_prec:.4f}, recall={best_rec:.4f}, "
        f"F1={best_f1:.4f}, coverage={best_cov:.2%}"
    )

    return best_theta


def calibrate_threshold_from_indices(df, model, cfg, scaler, feature_cols, calib_indices):
    """
    用验证集或训练集的 buy 信号 p_good 分布校准阈值。
    p_good 越大越好。

    desired_trade_ratio=0.30 表示执行 p_good 最高的 30%。
    """
    pred_df = predict_for_indices(
        df, model, cfg, scaler, feature_cols, calib_indices
    )

    if len(pred_df) == 0:
        return 0.5

    buy_pred = pred_df[pred_df['signal'] == 'buy'].copy()

    if len(buy_pred) == 0:
        return 0.5

    q = 1.0 - cfg.desired_trade_ratio
    theta = float(np.quantile(buy_pred['p_good'].values, q))

    theta = max(0.0, min(1.0, theta - 1e-8))

    print("\n===== p_good 阈值校准 =====")
    print(f"校准期 buy 信号数: {len(buy_pred)}")
    print(
        f"p_good min/median/max: "
        f"{buy_pred['p_good'].min():.4f} / "
        f"{buy_pred['p_good'].median():.4f} / "
        f"{buy_pred['p_good'].max():.4f}"
    )
    print(f"目标执行比例: {cfg.desired_trade_ratio:.2%}, p_good threshold={theta:.4f}")

    return theta


def calc_sharpe(equity_curve, periods_per_year=252):
    eq = np.array(equity_curve, dtype=float)

    if len(eq) < 3:
        return 0.0

    if np.any(eq[:-1] <= 0):
        return 0.0

    returns = eq[1:] / eq[:-1] - 1.0
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std <= 1e-12:
        return 0.0

    return np.sqrt(periods_per_year) * returns.mean() / std


def calc_max_drawdown(equity_curve):
    eq = np.array(equity_curve, dtype=float)

    if len(eq) == 0:
        return 0.0

    peak = np.maximum.accumulate(eq)
    peak = np.where(peak <= 1e-12, 1e-12, peak)

    dd = (eq - peak) / peak
    dd = dd[np.isfinite(dd)]

    if len(dd) == 0:
        return 0.0

    return dd.min()


def calc_total_return(equity_curve):
    """
    计算最终累计收益率：
    equity 从 1.0 开始，则最终收益率 = final_equity - 1
    """
    eq = np.array(equity_curve, dtype=float)

    if len(eq) == 0:
        return 0.0

    eq = eq[np.isfinite(eq)]

    if len(eq) == 0:
        return 0.0

    if abs(eq[0]) <= 1e-12:
        return 0.0

    return eq[-1] / eq[0] - 1.0


def backtest_buy_and_hold(df, start_loc=None, end_loc=None):
    """
    Buy and Hold 基线：
    在测试期起点用全部资金买入并一直持有到测试期结束。

    返回值：
    - equity list，长度与测试期日期数一致；
    - 第一个交易日 equity = 1.0；
    """
    if start_loc is None:
        start_loc = 0

    if end_loc is None:
        end_loc = len(df)

    start_loc = max(0, start_loc)
    end_loc = min(end_loc, len(df))

    if end_loc <= start_loc:
        return [1.0]

    prices = df['close'].iloc[start_loc:end_loc].values.astype(float)

    if len(prices) == 0:
        return [1.0]

    first_price = prices[0]

    if not np.isfinite(first_price) or first_price <= 1e-12:
        return np.ones(len(prices), dtype=float).tolist()

    equity = prices / first_price
    equity = np.where(np.isfinite(equity), equity, 1.0)

    return equity.tolist()


def align_equity_to_dates(equity_curve, target_len):
    """
    将不同回测函数返回的 equity 长度对齐到日期长度。

    当前 backtest_no_filter / apply_filter_and_trade 返回:
    - 初始 equity 一个点
    - 每个交易日结束后 equity 一个点
    因此通常长度 = 日期数 + 1

    Buy and Hold 返回:
    - 每个交易日一个点
    因此通常长度 = 日期数
    """
    eq = np.array(equity_curve, dtype=float)

    if target_len <= 0:
        return np.array([], dtype=float)

    if len(eq) == 0:
        return np.ones(target_len, dtype=float)

    # 如果长度比日期多 1，去掉初始点
    if len(eq) == target_len + 1:
        eq = eq[1:]

    # 如果刚好一致，直接使用
    elif len(eq) == target_len:
        pass

    # 如果过长，截断
    elif len(eq) > target_len:
        eq = eq[-target_len:]

    # 如果过短，用最后一个值补齐
    else:
        pad_value = eq[-1] if len(eq) > 0 and np.isfinite(eq[-1]) else 1.0
        pad = np.full(target_len - len(eq), pad_value, dtype=float)
        eq = np.concatenate([eq, pad], axis=0)

    eq = np.where(np.isfinite(eq), eq, 1.0)

    return eq


def build_equity_curve_df(
        code,
        df,
        start_loc,
        end_loc,
        test_signal_locs,
        equity_ctaml,
        equity_signal,
        equity_buy_hold,
        equity_oracle=None
):
    """
    构造逐日收益率曲线 DataFrame，方便后续画图。

    包含：
    - 模型过滤 CTAML
    - 原始信号无过滤 signal
    - Buy and Hold
    - Oracle 标签过滤，可选
    """
    start_loc = max(0, start_loc)
    end_loc = min(end_loc, len(df))

    if end_loc <= start_loc:
        return pd.DataFrame()

    dates = df.index[start_loc:end_loc]
    target_len = len(dates)

    eq_ctaml = align_equity_to_dates(equity_ctaml, target_len)
    eq_signal = align_equity_to_dates(equity_signal, target_len)
    eq_bh = align_equity_to_dates(equity_buy_hold, target_len)

    if equity_oracle is not None:
        eq_oracle = align_equity_to_dates(equity_oracle, target_len)
    else:
        eq_oracle = np.full(target_len, np.nan, dtype=float)

    test_signal_locs = set(test_signal_locs) if test_signal_locs is not None else set()
    locs = list(range(start_loc, end_loc))

    curve_df = pd.DataFrame({
        'code': code,
        'time': dates,
        'close': df['close'].iloc[start_loc:end_loc].values,
        'raw_signal': df['signal'].iloc[start_loc:end_loc].values,
        'is_test_signal': [loc in test_signal_locs for loc in locs],

        'equity_ctaml': eq_ctaml,
        'return_ctaml': eq_ctaml - 1.0,

        'equity_signal': eq_signal,
        'return_signal': eq_signal - 1.0,

        'equity_buy_hold': eq_bh,
        'return_buy_hold': eq_bh - 1.0,

        'equity_oracle': eq_oracle,
        'return_oracle': eq_oracle - 1.0
    })

    return curve_df


def save_equity_curve_df(curve_df, cfg, code):
    """
    保存单只股票逐日收益率曲线。
    """
    if curve_df is None or len(curve_df) == 0:
        return None

    curve_dir = os.path.join(cfg.results_dir, "equity_curves")
    os.makedirs(curve_dir, exist_ok=True)

    out_path = os.path.join(
        curve_dir,
        f"equity_{cfg.run_tag}_{code}.csv"
    )

    curve_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    return out_path

def analyze_prediction_vs_label(df, buy_records, good_threshold=0.4):
    if not buy_records:
        print("无 buy_records 可分析")
        return None

    rec = pd.DataFrame(buy_records).copy()
    rec['time'] = pd.to_datetime(rec['time'])

    label_df = df[['D_label']].copy()
    label_df = label_df.reset_index().rename(columns={'index': 'time'})
    label_df['time'] = pd.to_datetime(label_df['time'])

    merged = rec.merge(label_df, on='time', how='left')
    merged['true_good'] = merged['D_label'] <= good_threshold

    print("\n===== 预测 vs 真实标签诊断 =====")
    print(f"buy 信号数: {len(merged)}")
    print(f"真实好信号数 D_label<={good_threshold}: {merged['true_good'].sum()}")
    print(f"模型执行数: {merged['executed'].sum()}")

    if merged['executed'].sum() > 0:
        exec_df = merged[merged['executed']]
        print(f"执行信号中的真实好信号数: {exec_df['true_good'].sum()} / {len(exec_df)}")
        print(f"执行信号真实好信号占比: {exec_df['true_good'].mean():.2%}")
        print(f"执行信号平均 D_label: {exec_df['D_label'].mean():.4f}")

    skip_df = merged[~merged['executed']]
    if len(skip_df) > 0:
        print(f"跳过信号中的真实好信号数: {skip_df['true_good'].sum()} / {len(skip_df)}")
        print(f"跳过信号真实好信号占比: {skip_df['true_good'].mean():.2%}")
        print(f"跳过信号平均 D_label: {skip_df['D_label'].mean():.4f}")

    if merged['D_label'].notna().sum() > 3:
        corr_d = merged[['d_mid', 'D_label']].corr(method='spearman').iloc[0, 1]
        print(f"d_mid 与 D_label 的 Spearman 相关: {corr_d:.4f}")

        if 'p_good' in merged.columns:
            corr_p = merged[['p_good', 'D_label']].corr(method='spearman').iloc[0, 1]
            print(f"p_good 与 D_label 的 Spearman 相关: {corr_p:.4f}，理论上应为负相关")

    sort_col = 'p_good' if 'p_good' in merged.columns else 'd_mid'
    ascending = False if sort_col == 'p_good' else True

    cols = ['time', 'price', 'd_mid']
    if 'p_good' in merged.columns:
        cols.append('p_good')
    cols += ['D_label', 'true_good', 'executed']

    print(f"\n按 {sort_col} 排序：")
    print(
        merged[cols]
        .sort_values(sort_col, ascending=ascending)
        .to_string(index=False)
    )

    return merged


# ================== 8. 主程序（示例） ==================
def main():
    set_seed(cfg.seed)
    if cfg.save_outputs:
        os.makedirs(cfg.results_dir, exist_ok=True)

    if cfg.strategy_name == 'BOLL_RSI':
        signal_dir = "./QuantData/signals/BOLL_RSI"
    elif cfg.strategy_name == 'MACD_KDJ':
        signal_dir = "./QuantData/signals/MACD_KDJ"
    else:
        raise ValueError(f"未知策略: {cfg.strategy_name}")

    if cfg.market_name == 'A':
        stock_codes = [
            "002311", "002493", "600588", "002049", "000977", "000617", "601111", "600233", "600115", "000661",
            "601006", "002475", "000876", "601600", "002179", "601818", "601117", "600438", "600031", "600089",
            "000568", "002304", "002050", "601318", "600111", "600426", "601618", "600176", "600893", "600104"
        ]
        ohlcv_paths = [
            f"./QuantData/ohlcv/bar_A_{code}_d.csv"
            for code in stock_codes
        ]
        signal_paths = [
            f"{signal_dir}/A_{code}_d.csv"
            for code in stock_codes
        ]
    elif cfg.market_name == 'US':
        stock_codes = [
            "NFLX", "PYPL", "CSCO", "IBM", "BAC", "GS", "C", "T", "PFE", "CAT",
            "F", "AXP", "AIG", "MMM", "LOW", "BBY", "WBA", "CMG", "UAL", "EA",
            "PM", "HPQ", "MNST", "COF", "SYY", "AAPL", "MSFT", "TSLA", "WMT", "KO"
        ]
        ohlcv_paths = [
            f"./QuantData/ohlcv/bar_USA_{code}_d.csv"
            for code in stock_codes
        ]
        signal_paths = [
            f"{signal_dir}/USA_{code}_d.csv"
            for code in stock_codes
        ]
    else:
        raise ValueError(f"未知市场: {cfg.market_name}")

    # ---------- 第一阶段：收集所有股票的训练集特征，用于拟合全局 scaler ----------
    global_train_features = []  # 存储每只股票训练期的特征数组
    stock_data_info = []  # 存储每只股票的处理信息，供第二阶段使用

    for code, sig_path, ohlcv_path in zip(stock_codes, signal_paths, ohlcv_paths):
        print(f"处理股票: {code}")
        cfg.signal_file = sig_path
        cfg.ohlcv_file = ohlcv_path

        # 加载并标注（标注使用全量数据）
        df, feature_cols = load_and_prepare(sig_path, ohlcv_path)
        df, xi_series, sigma_series, extrema_info = compute_evt_distance_xi(df, cfg)

        train_indices, val_indices, test_indices = split_signals_train_val_test(
            df, cfg, train_ratio=0.6, val_ratio=0.2
        )

        # 标签诊断
        print_label_diagnostics(code, df, train_indices, prefix="训练期")
        print_label_diagnostics(code, df, val_indices, prefix="验证期")
        print_label_diagnostics(code, df, test_indices, prefix="测试期")
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"股票 {code} 训练或测试信号不足，跳过")
            continue

        # 确定训练期的时间范围（最后一个训练信号的时间）
        train_times = [df.index[idx] for idx in train_indices]  # 训练信号对应的时间戳
        end_train_time = max(train_times)  # 训练期结束时间

        # 提取该股票训练期的所有特征行（用于 fit 全局 scaler）
        train_mask = df.index <= end_train_time
        train_features_stock = df[feature_cols].loc[train_mask].values
        global_train_features.append(train_features_stock)

        # 保存信息供第二阶段使用
        stock_data_info.append((
            code, df, feature_cols, extrema_info,
            train_indices, val_indices, test_indices, end_train_time
        ))

    if len(global_train_features) == 0:
        print("没有可用的训练数据")
        return

    # 用所有股票的训练数据拟合全局 scaler
    all_train_data = np.concatenate(global_train_features, axis=0)
    global_scaler = StandardScaler()
    global_scaler.fit(all_train_data)
    print(f"全局 scaler 已拟合，训练数据总行数: {len(all_train_data)}")

    # ---------- 第二阶段：用全局 scaler 创建 Dataset ----------
    all_train_datasets = []
    all_val_datasets = []
    test_info = []

    for code, df, feature_cols, extrema_info, train_indices, val_indices, test_indices, _ in stock_data_info:
        train_ds = SignalDataset(
            df, feature_cols, cfg, extrema_info,
            scaler=global_scaler,
            signal_indices=train_indices,
            signal_filter_type=cfg.train_signal_type
        )

        val_ds = SignalDataset(
            df, feature_cols, cfg, extrema_info,
            scaler=global_scaler,
            signal_indices=val_indices,
            signal_filter_type=cfg.train_signal_type
        )

        test_ds = SignalDataset(
            df, feature_cols, cfg, extrema_info,
            scaler=global_scaler,
            signal_indices=test_indices,
            signal_filter_type=cfg.train_signal_type
        )

        if len(train_ds) == 0:
            print(f"股票 {code} 有 train_indices，但构造 train_ds 后样本数为 0，跳过")
            continue

        all_train_datasets.append(train_ds)

        if len(val_ds) > 0:
            all_val_datasets.append(val_ds)

        # 这里额外保存 train_indices 和 test_indices，方便阈值校准和严格测试集回测
        test_info.append((
            code, df, test_ds, global_scaler, feature_cols,
            extrema_info, train_indices, val_indices, test_indices
        ))

    if len(all_train_datasets) == 0:
        print("所有股票构造后的训练 Dataset 都为空，终止")
        return
    # 合并所有股票训练数据
    combined_train_ds = torch.utils.data.ConcatDataset(all_train_datasets)

    if cfg.use_balanced_sampler:
        sampler = build_balanced_sampler(combined_train_ds)
    else:
        sampler = None

    if sampler is not None:
        train_loader = DataLoader(
            combined_train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=False
        )
    else:
        train_loader = DataLoader(
            combined_train_ds,
            batch_size=cfg.batch_size,
            shuffle=True
        )

    if len(all_val_datasets) > 0:
        combined_val_ds = torch.utils.data.ConcatDataset(all_val_datasets)
        val_loader = DataLoader(combined_val_ds, batch_size=cfg.batch_size, shuffle=False)
    else:
        val_loader = None

    # 初始化模型
    if cfg.encoder_type == 'BiLSTM':
        encoder = BiLSTMEncoder(input_dim=len(feature_cols), d_model=cfg.d_model)
    else:
        encoder = InformerEncoder(input_dim=len(feature_cols), d_model=cfg.d_model,
                                  seq_len=cfg.window_size, n_heads=cfg.n_heads)
    model = CTAML(encoder, d_model=cfg.d_model, n_heads=cfg.n_heads,
                  proj_dim=cfg.proj_dim, quantiles=cfg.quantiles,
                  use_taa=cfg.use_taa).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 训练
    best_score = -float('inf')
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, cfg)

        if val_loader is not None:
            val_metrics = eval_classifier_metrics(model, val_loader, cfg)
            val_loss = val_metrics['loss']

            print(
                f"Epoch {epoch + 1}/{cfg.epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val AUC: {val_metrics['auc'] if np.isfinite(val_metrics['auc']) else np.nan:.4f}, "
                f"Val AP: {val_metrics['ap'] if np.isfinite(val_metrics['ap']) else np.nan:.4f}, "
                f"p_good median: {val_metrics['p_median']:.4f}, "
                f"good_mean: {val_metrics['good_mean']:.4f}, "
                f"bad_mean: {val_metrics['bad_mean']:.4f}, "
                f"good_count: {val_metrics['good_count']}/{val_metrics['n']}"
            )

            if cfg.early_stop_metric == 'val_auc' and np.isfinite(val_metrics['auc']):
                monitor_score = val_metrics['auc']
                improved = monitor_score > best_score
            elif cfg.early_stop_metric == 'val_ap' and np.isfinite(val_metrics['ap']):
                monitor_score = val_metrics['ap']
                improved = monitor_score > best_score
            else:
                monitor_score = -val_loss
                improved = val_loss < best_loss

        else:
            print(f"Epoch {epoch + 1}/{cfg.epochs}, Train Loss: {train_loss:.4f}")
            monitor_score = -train_loss
            improved = train_loss < best_loss

        if improved:
            if val_loader is not None and cfg.early_stop_metric in ['val_auc', 'val_ap']:
                best_score = monitor_score
            else:
                best_loss = -monitor_score if val_loader is None else val_loss

            patience_counter = 0
            torch.save(model.state_dict(), "best_ctaml.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("早停触发")
                break

    # 加载最佳模型用于测试
    model_eval = CTAML(encoder, d_model=cfg.d_model, n_heads=cfg.n_heads,
                       proj_dim=cfg.proj_dim, quantiles=cfg.quantiles,
                       use_taa=cfg.use_taa).to(cfg.device)
    model_eval.load_state_dict(torch.load("best_ctaml.pth", map_location=cfg.device))
    model_eval.eval()
    # ---------- 全局验证集阈值校准 ----------
    global_val_preds = []

    for code, df, test_ds, scaler, feature_cols, extrema_info, train_indices, val_indices, test_indices in test_info:
        if len(val_indices) == 0:
            continue

        pred_val = predict_for_indices(
            df, model_eval, cfg, scaler, feature_cols, val_indices
        )

        pred_val = attach_labels_to_prediction(
            df, pred_val, good_threshold=cfg.good_D_threshold
        )

        if len(pred_val) > 0:
            pred_val['code'] = code
            global_val_preds.append(pred_val)

    if len(global_val_preds) > 0:
        global_val_pred_df = pd.concat(global_val_preds, axis=0, ignore_index=True)
    else:
        global_val_pred_df = pd.DataFrame()

    global_threshold = calibrate_global_threshold_from_val(global_val_pred_df, cfg)
    print(f"\n最终全局 p_good threshold = {global_threshold:.4f}")

    # 对每只股票单独回测并汇总指标
    summary = []

    # 保存所有股票的逐日收益率曲线，最后合并输出
    all_equity_curve_dfs = []

    for code, df, test_ds, scaler, feature_cols, extrema_info, train_indices, val_indices, test_indices in test_info:
        if len(test_indices) == 0:
            print(f"股票 {code} 测试信号为空，跳过")
            continue

        test_start_loc = max(min(test_indices), cfg.window_size)
        test_end_loc = len(df)

        # 1. 用训练期预测分布校准阈值
        # 使用全局验证集阈值，避免单股票验证 buy 数太少导致阈值漂移
        threshold = global_threshold

        # 2. 过滤后测试集回测
        equity_f, buy_records = apply_filter_and_trade(
            df, model_eval, cfg, scaler, feature_cols,
            signal_locs=test_indices,
            start_loc=test_start_loc,
            end_loc=test_end_loc,
            threshold=threshold
        )

        # 3. 无过滤测试集基线
        equity_b = backtest_no_filter(
            df, cfg,
            signal_locs=test_indices,
            start_loc=test_start_loc,
            end_loc=test_end_loc
        )

        # 4. Oracle 标签回测：诊断 D_label 是否本身有效
        equity_o, oracle_records = backtest_oracle_label(
            df, cfg,
            signal_locs=test_indices,
            start_loc=test_start_loc,
            end_loc=test_end_loc,
            threshold=cfg.oracle_D_threshold
        )

        # 5. Buy and Hold 基线：测试期起点买入并持有到结束
        equity_h = backtest_buy_and_hold(
            df,
            start_loc=test_start_loc,
            end_loc=test_end_loc
        )

        # 6. 打印模型预测分布
        if buy_records:
            df_rec = pd.DataFrame(buy_records)
            print(f"\n===== 股票 {code} 测试集买入信号 d_mid 分析 =====")
            print(f"测试集买入信号数: {len(df_rec)}")
            print(f"threshold: {threshold:.4f}")
            print(f"d_mid 最小值: {df_rec['d_mid'].min():.4f}")
            print(f"d_mid 最大值: {df_rec['d_mid'].max():.4f}")
            print(f"d_mid 中位数: {df_rec['d_mid'].median():.4f}")
            print(f"p_good 最小值: {df_rec['p_good'].min():.4f}")
            print(f"p_good 最大值: {df_rec['p_good'].max():.4f}")
            print(f"p_good 中位数: {df_rec['p_good'].median():.4f}")

            executed_count = int(df_rec['executed'].sum())
            print(f"实际执行买入信号数: {executed_count}")

            if executed_count > 0:
                print("执行的买入信号详情:")
                print(df_rec[df_rec['executed']][['time', 'price', 'd_mid', 'p_good', 'threshold']].to_string(
                    index=False))
        else:
            print(f"股票 {code} 测试集无买入信号记录")

        # 6. Oracle 打印
        if oracle_records:
            df_oracle = pd.DataFrame(oracle_records)
            oracle_exec = int(df_oracle['executed'].sum())
            print(f"\n===== 股票 {code} Oracle 标签过滤诊断 =====")
            print(f"Oracle buy 信号数: {len(df_oracle)}")
            print(f"Oracle 执行数: {oracle_exec}")
            print(f"Oracle D_threshold: {cfg.oracle_D_threshold:.4f}")
            print(f"D_label min/median/max: "
                  f"{df_oracle['D_label'].min():.4f} / "
                  f"{df_oracle['D_label'].median():.4f} / "
                  f"{df_oracle['D_label'].max():.4f}")

        sharpe_f = calc_sharpe(equity_f)
        mdd_f = calc_max_drawdown(equity_f)
        ret_f = calc_total_return(equity_f)

        sharpe_b = calc_sharpe(equity_b)
        mdd_b = calc_max_drawdown(equity_b)
        ret_b = calc_total_return(equity_b)

        sharpe_o = calc_sharpe(equity_o)
        mdd_o = calc_max_drawdown(equity_o)
        ret_o = calc_total_return(equity_o)

        sharpe_h = calc_sharpe(equity_h)
        mdd_h = calc_max_drawdown(equity_h)
        ret_h = calc_total_return(equity_h)

        # 构造并保存该股票逐日收益率曲线
        curve_df = build_equity_curve_df(
            code=code,
            df=df,
            start_loc=test_start_loc,
            end_loc=test_end_loc,
            test_signal_locs=test_indices,
            equity_ctaml=equity_f,
            equity_signal=equity_b,
            equity_buy_hold=equity_h,
            equity_oracle=equity_o
        )

        if cfg.save_outputs and cfg.save_equity_curves and len(curve_df) > 0:
            curve_path = save_equity_curve_df(curve_df, cfg, code)
            print(f"收益率曲线已保存: {curve_path}")

        if len(curve_df) > 0:
            all_equity_curve_dfs.append(curve_df)

        buy_count = len(buy_records)
        exec_count = int(pd.DataFrame(buy_records)['executed'].sum()) if buy_records else 0
        coverage = exec_count / buy_count if buy_count > 0 else 0.0

        summary.append((
            code,
            sharpe_f, mdd_f, ret_f,
            sharpe_b, mdd_b, ret_b,
            sharpe_o, mdd_o, ret_o,
            sharpe_h, mdd_h, ret_h,
            buy_count, exec_count, coverage
        ))

        print(
            f"\n股票 {code}: "
            f"CTAML过滤 夏普={sharpe_f:.4f}, MDD={mdd_f:.4%}, 最终收益={ret_f:.4%} | "
            f"无过滤 夏普={sharpe_b:.4f}, MDD={mdd_b:.4%}, 最终收益={ret_b:.4%} | "
            f"Oracle标签 夏普={sharpe_o:.4f}, MDD={mdd_o:.4%}, 最终收益={ret_o:.4%} | "
            f"Buy&Hold 夏普={sharpe_h:.4f}, MDD={mdd_h:.4%}, 最终收益={ret_h:.4%}"
        )

        analyze_prediction_vs_label(df, buy_records, good_threshold=cfg.oracle_D_threshold)

    if summary:
        avg_sharpe_f = np.mean([x[1] for x in summary])
        avg_mdd_f = np.mean([x[2] for x in summary])
        avg_ret_f = np.mean([x[3] for x in summary])

        avg_sharpe_b = np.mean([x[4] for x in summary])
        avg_mdd_b = np.mean([x[5] for x in summary])
        avg_ret_b = np.mean([x[6] for x in summary])

        avg_sharpe_o = np.mean([x[7] for x in summary])
        avg_mdd_o = np.mean([x[8] for x in summary])
        avg_ret_o = np.mean([x[9] for x in summary])

        avg_sharpe_h = np.mean([x[10] for x in summary])
        avg_mdd_h = np.mean([x[11] for x in summary])
        avg_ret_h = np.mean([x[12] for x in summary])

        total_buy = np.sum([x[13] for x in summary])
        total_exec = np.sum([x[14] for x in summary])
        avg_coverage = np.mean([x[15] for x in summary])
        global_coverage = total_exec / total_buy if total_buy > 0 else 0.0

        print("\n===== 汇总结果 =====")
        print(
            f"CTAML过滤: 平均夏普={avg_sharpe_f:.4f}, "
            f"平均MDD={avg_mdd_f:.4%}, 平均最终收益={avg_ret_f:.4%}"
        )
        print(
            f"无过滤:     平均夏普={avg_sharpe_b:.4f}, "
            f"平均MDD={avg_mdd_b:.4%}, 平均最终收益={avg_ret_b:.4%}"
        )
        print(
            f"Oracle标签: 平均夏普={avg_sharpe_o:.4f}, "
            f"平均MDD={avg_mdd_o:.4%}, 平均最终收益={avg_ret_o:.4%}"
        )
        print(
            f"Buy&Hold:   平均夏普={avg_sharpe_h:.4f}, "
            f"平均MDD={avg_mdd_h:.4%}, 平均最终收益={avg_ret_h:.4%}"
        )
        print(
            f"CTAML交易覆盖率: 平均覆盖率={avg_coverage:.2%}, "
            f"全局覆盖率={global_coverage:.2%}, 执行/买入={int(total_exec)}/{int(total_buy)}"
        )

        if cfg.save_outputs:
            summary_df = pd.DataFrame(
                summary,
                columns=[
                    'code',

                    'sharpe_ctaml', 'mdd_ctaml', 'return_ctaml',

                    'sharpe_baseline', 'mdd_baseline', 'return_baseline',

                    'sharpe_oracle', 'mdd_oracle', 'return_oracle',

                    'sharpe_buy_hold', 'mdd_buy_hold', 'return_buy_hold',

                    'buy_count', 'exec_count', 'coverage'
                ]
            )

            summary_df['delta_sharpe_ctaml_vs_base'] = (
                summary_df['sharpe_ctaml'] - summary_df['sharpe_baseline']
            )
            summary_df['delta_sharpe_oracle_vs_base'] = (
                summary_df['sharpe_oracle'] - summary_df['sharpe_baseline']
            )
            summary_df['mdd_improve_ctaml_vs_base'] = (
                summary_df['mdd_ctaml'] - summary_df['mdd_baseline']
            )
            summary_df['mdd_improve_oracle_vs_base'] = (
                summary_df['mdd_oracle'] - summary_df['mdd_baseline']
            )
            summary_df['delta_return_ctaml_vs_base'] = (
                summary_df['return_ctaml'] - summary_df['return_baseline']
            )
            summary_df['delta_return_oracle_vs_base'] = (
                summary_df['return_oracle'] - summary_df['return_baseline']
            )
            summary_df['delta_return_ctaml_vs_buy_hold'] = (
                summary_df['return_ctaml'] - summary_df['return_buy_hold']
            )
            summary_df['delta_return_base_vs_buy_hold'] = (
                summary_df['return_baseline'] - summary_df['return_buy_hold']
            )

            out_path = os.path.join(
                cfg.results_dir,
                f"summary_{cfg.run_tag}.csv"
            )
            summary_df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"summary 已保存: {out_path}")

            print("\n===== 改善统计 =====")
            print(
                f"CTAML胜过无过滤的股票数: "
                f"{(summary_df['delta_sharpe_ctaml_vs_base'] > 0).sum()} / {len(summary_df)}"
            )
            print(
                f"Oracle胜过无过滤的股票数: "
                f"{(summary_df['delta_sharpe_oracle_vs_base'] > 0).sum()} / {len(summary_df)}"
            )
            print(
                f"CTAML平均夏普提升: {summary_df['delta_sharpe_ctaml_vs_base'].mean():.4f}"
            )
            print(
                f"Oracle平均夏普提升: {summary_df['delta_sharpe_oracle_vs_base'].mean():.4f}"
            )
            print(
                f"CTAML平均MDD改善: {summary_df['mdd_improve_ctaml_vs_base'].mean():.4%}"
            )
            print(
                f"Oracle平均MDD改善: {summary_df['mdd_improve_oracle_vs_base'].mean():.4%}"
            )
            print(
                f"CTAML平均最终收益提升: {summary_df['delta_return_ctaml_vs_base'].mean():.4%}"
            )
            print(
                f"Oracle平均最终收益提升: {summary_df['delta_return_oracle_vs_base'].mean():.4%}"
            )
            print(
                f"CTAML胜过Buy&Hold的股票数: "
                f"{(summary_df['delta_return_ctaml_vs_buy_hold'] > 0).sum()} / {len(summary_df)}"
            )
            print(
                f"原始信号胜过Buy&Hold的股票数: "
                f"{(summary_df['delta_return_base_vs_buy_hold'] > 0).sum()} / {len(summary_df)}"
            )
            # 保存所有股票合并后的逐日收益率曲线
            if cfg.save_equity_curves and len(all_equity_curve_dfs) > 0:
                all_curve_df = pd.concat(all_equity_curve_dfs, axis=0, ignore_index=True)

                curve_dir = os.path.join(cfg.results_dir, "equity_curves")
                os.makedirs(curve_dir, exist_ok=True)

                all_curve_path = os.path.join(
                    curve_dir,
                    f"equity_{cfg.run_tag}_ALL.csv"
                )

                all_curve_df.to_csv(all_curve_path, index=False, encoding='utf-8-sig')
                print(f"全部股票收益率曲线已保存: {all_curve_path}")

if __name__ == "__main__":
    main()
