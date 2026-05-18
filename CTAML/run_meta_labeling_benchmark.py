"""
对比尾部感知元标签 (CTAML)：完整实验代码
根据论文"Contrastive Tail-Aware Meta-Labeling: Self-Supervised Extreme Region Distance Field Learning for Trading Signal Filtering"
目标期刊：ESWA / EAAI / ASOC / Neurocomputing / KBS / Information Sciences

依赖：pandas, numpy, torch, scipy, sklearn, matplotlib, tqdm

本文实验框架支持任意标注函数，只要求标注器输出连续诊断标签 label_value 和二分类元标签 meta_label。
主实验可选择 ERDF-EVT、固定局部极值、未来收益、Triple Barrier 等不同标签，并统一在 Original Signal、Triple Barrier + XGB 和 Buy-and-Hold 基线下评估。
"""
# 导入必要的科学计算、数据处理和深度学习库
import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from scipy.stats import wilcoxon
from label_methods import build_labeler, print_label_diagnostics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt

# XGBoost 可选依赖：如果没安装 xgboost，则自动回退到 sklearn 的 HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')


# ================== 配置 ==================
class Config:
    # ================= 数据 =================
    signal_file = ""
    ohlcv_file = ""

    # ================= 通用序列窗口参数 =================
    window_size = 160
    future_T = 60
    purge_gap = 5

    # ================= 标注方法 API =================
    # 可选：
    # 'erdf_evt'
    # 'fixed_extrema'
    # 'future_return'
    # 'triple_barrier_meta'
    # up20pct
    label_method = 'triple_barrier_meta'

    # 所有标注方法都必须输出这两个标准列
    label_value_col = 'label_value'
    meta_label_col = 'meta_label'

    # 模型训练的信号类型。当前框架默认只做 buy-side filtering。
    train_signal_type = 'buy'

    # 对 ERDF / fixed_extrema 这类“距离越小越好”的标签：
    # D <= good_label_threshold 视为好 buy 信号。
    good_label_threshold = 0.4

    # 对 future_return 标签：
    # future return >= future_return_threshold 视为好 buy 信号。
    future_return_threshold = 0.2

    # ================= ERDF-EVT 标签参数 =================
    # 仅当 label_method='erdf_evt' 时使用
    kappa = 1.5
    evt_rolling = 252
    evt_quantile = 0.90
    extrema_win = 3

    # ================= 固定局部极值标签参数 =================
    # 仅当 label_method='fixed_extrema' 时使用
    fixed_extrema_win = 3

    # ================= 序列 Meta-Label 模型 =================
    encoder_type = 'BiLSTM'
    d_model = 64

    batch_size = 32
    lr = 5e-4
    epochs = 120
    patience = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_balanced_sampler = True
    early_stop_metric = 'val_auc'

    # ================= 阈值校准 =================
    desired_trade_ratio = 0.30
    threshold_method = 'global_f1_constrained'
    fallback_pgood_threshold = 0.5
    min_val_coverage = 0.20
    max_val_coverage = 0.45

    # ================= Triple Barrier + XGB 外部基线 =================
    enable_triple_barrier_xgb = True
    tb_horizon = 260
    tb_pt_atr = 1.5
    tb_sl_atr = 1.0
    xgb_threshold_method = 'global_f1_constrained'

    # ================= 多随机种子 =================
    run_multi_seed = False
    seeds = [11, 22, 33, 42, 55]
    seed = 42

    # ================= 交易成本敏感性 =================
    cost_bps_list = [0, 5, 10, 20]

    # ================= 实验管理 =================
    strategy_name = 'MACD_KDJ'   # 'BOLL_RSI' / 'MACD_KDJ'
    market_name = 'US'            # 'A' / 'US'
    run_tag = 'TB_META_MACD_KDJ_US'

    results_dir = './ctaml_results'
    save_outputs = True
    save_equity_curves = True

    # ================= 画图 =================
    save_plots = True
    plot_dir = './ctaml_results/figures'
    plot_top_n = 5
    plot_dpi = 300

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
def compute_triple_barrier_labels(df, signal_indices, cfg):
    """
    对 buy 信号构造 Triple Barrier 标签：
    y=1: 未来 horizon 内先触及上障碍
    y=0: 先触及下障碍或未触及上障碍

    上障碍 = close + tb_pt_atr * ATR
    下障碍 = close - tb_sl_atr * ATR
    """
    labels = {}

    n = len(df)

    for i in signal_indices:
        if i >= n:
            continue

        if df['signal'].iloc[i] != 'buy':
            continue

        price0 = df['close'].iloc[i]
        atr0 = df['atr'].iloc[i]

        if not np.isfinite(price0) or not np.isfinite(atr0) or atr0 <= 1e-12:
            continue

        upper = price0 + cfg.tb_pt_atr * atr0
        lower = price0 - cfg.tb_sl_atr * atr0

        end = min(n, i + cfg.tb_horizon + 1)
        y = 0

        for j in range(i + 1, end):
            high_j = df['high'].iloc[j]
            low_j = df['low'].iloc[j]

            hit_upper = high_j >= upper
            hit_lower = low_j <= lower

            if hit_upper and hit_lower:
                # 保守处理：同一天上下都触发时视为失败
                y = 0
                break
            elif hit_upper:
                y = 1
                break
            elif hit_lower:
                y = 0
                break

        labels[i] = y

    return labels


# ================== 3. 数据集与对比学习准备 ==================
class SignalDataset(Dataset):
    """
    通用买入信号过滤数据集。

    输入：
    - x_win: 信号日前 window_size 个时间步的特征窗口；
    - meta_label: 任意标注方法生成的二分类标签；
    - abs_ret: 保留给未来 tail-aware 或波动率模块使用。

    要求：
    df 中必须已经存在 cfg.meta_label_col。
    """

    def __init__(self, df, feature_cols, cfg, scaler=None,
                 signal_indices=None, signal_filter_type='buy'):
        self.df = df
        self.feature_cols = feature_cols
        self.features = df[feature_cols].values.astype(np.float32)

        self.label_col = cfg.meta_label_col

        if self.label_col not in df.columns:
            raise ValueError(
                f"SignalDataset 找不到标签列 {self.label_col}. "
                f"请先调用 labeler.transform(df, cfg) 生成标签。"
            )

        self.labels = df[self.label_col].values.astype(float)
        self.window = cfg.window_size
        self.cfg = cfg

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
            if i + cfg.future_T >= len(df):
                continue
            if np.isnan(self.labels[i]):
                continue

            sig = self.df['signal'].iloc[i]

            if signal_filter_type is not None and sig != signal_filter_type:
                continue

            self.signal_indices.append(i)

        self.good_labels = np.array([
            float(self.labels[i]) for i in self.signal_indices
        ], dtype=np.float32)

    def get_good_labels(self):
        return self.good_labels

    def __len__(self):
        return len(self.signal_indices)

    def __getitem__(self, idx):
        i = self.signal_indices[idx]

        x_win = self.features[i - self.window:i]
        good_label = float(self.labels[i])
        abs_ret = self.df['abs_returns'].values[i - self.window:i].astype(np.float32)

        return (
            torch.tensor(x_win, dtype=torch.float32),
            torch.tensor(good_label, dtype=torch.float32),
            torch.tensor(abs_ret, dtype=torch.float32)
        )


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


def make_tabular_features_for_signal(df, feature_cols, i, cfg):
    """
    给 XGB 使用的表格特征。
    使用信号日前 window 的：
    - 最后一天特征
    - 均值
    - 标准差
    """
    if i < cfg.window_size:
        return None

    win = df[feature_cols].iloc[i - cfg.window_size:i].values.astype(float)

    if not np.all(np.isfinite(win)):
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

    last_feat = win[-1]
    mean_feat = win.mean(axis=0)
    std_feat = win.std(axis=0)

    feat = np.concatenate([last_feat, mean_feat, std_feat], axis=0)

    return feat


def build_xgb_dataset(df, feature_cols, indices, labels_dict, cfg):
    X = []
    y = []
    locs = []

    for i in indices:
        if i not in labels_dict:
            continue
        if df['signal'].iloc[i] != 'buy':
            continue

        feat = make_tabular_features_for_signal(df, feature_cols, i, cfg)
        if feat is None:
            continue

        X.append(feat)
        y.append(labels_dict[i])
        locs.append(i)

    if len(X) == 0:
        return np.empty((0, 1)), np.array([]), []

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), locs


# ================== 4. 模型定义 ==================
# BiLSTM 编码器（输出序列）
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            d_model // 2,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.proj(out)


class MetaLabelSequenceClassifier(nn.Module):
    """
    通用序列 Meta-Label 分类器。

    输入：
    - 信号日前 window_size 个交易日的特征窗口

    输出：
    - p_good 的 logit，即该 buy 信号是否值得执行
    """

    def __init__(self, input_dim, d_model=64):
        super().__init__()
        self.encoder = BiLSTMEncoder(input_dim=input_dim, d_model=d_model)
        self.head_good = nn.Linear(d_model, 1)

    def forward(self, x):
        h_seq = self.encoder(x)
        h = h_seq[:, -1, :]
        good_logit = self.head_good(h).squeeze(-1)
        return good_logit


# ================== 5. 损失函数 ==================
def weighted_bce_loss(logits, targets):
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


# ================== 6. 训练循环 ==================
def train_triple_barrier_xgb(all_train_x, all_train_y, seed=42):
    """
    传统 Meta-Labeling 基线：
    Triple Barrier 标签 + XGBoost 分类器。
    """
    if len(all_train_y) == 0 or len(np.unique(all_train_y)) < 2:
        print("XGB训练集类别不足，无法训练")
        return None

    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1
        )
    else:
        print("未检测到 xgboost，使用 HistGradientBoostingClassifier 回退")
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=15,
            random_state=seed
        )

    model.fit(all_train_x, all_train_y)

    return model


def predict_xgb_proba(model, X):
    if model is None or len(X) == 0:
        return np.array([])

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    # fallback
    p = model.predict(X)
    return np.asarray(p, dtype=float)


def train_epoch(model, dataloader, optimizer, cfg):
    model.train()
    total_loss = 0.0

    for x, good_label, abs_ret in dataloader:
        x = x.to(cfg.device)
        good_label = good_label.to(cfg.device)

        good_logit = model(x)
        loss = weighted_bce_loss(good_logit, good_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def eval_classifier_metrics(model, dataloader, cfg):
    model.eval()

    all_y = []
    all_p = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, good_label, abs_ret in dataloader:
            x = x.to(cfg.device)
            good_label = good_label.to(cfg.device)

            good_logit = model(x)
            loss = weighted_bce_loss(good_logit, good_label)

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
            if loc + cfg.future_T + cfg.purge_gap < min_val_loc
        ]

    # 防止验证样本未来窗口触碰测试集
    if test_locs:
        min_test_loc = min(test_locs)
        val_locs = [
            loc for loc in val_locs
            if loc + cfg.future_T + cfg.purge_gap < min_test_loc
        ]

    return train_locs, val_locs, test_locs


# ================== 7. 回测与决策 ==================
def apply_xgb_filter_and_trade(df, xgb_model, cfg, feature_cols,
                               signal_locs=None,
                               start_loc=None,
                               end_loc=None,
                               threshold=0.5,
                               cost_bps=0):
    cost_rate = bps_to_rate(cost_bps)

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
    records = []

    for i_loc in range(start_loc, end_loc):
        row = df.iloc[i_loc]
        idx = df.index[i_loc]

        if i_loc in signal_locs and row['signal'] != 'none':
            if row['signal'] == 'buy':
                feat = make_tabular_features_for_signal(df, feature_cols, i_loc, cfg)

                if feat is None or xgb_model is None:
                    p_good = 0.0
                else:
                    p_good = float(predict_xgb_proba(xgb_model, feat.reshape(1, -1))[0])

                executed = p_good >= threshold

                records.append({
                    'time': idx,
                    'price': row['close'],
                    'p_xgb': p_good,
                    'threshold': threshold,
                    'executed': executed
                })

                if executed and position == 0:
                    position, capital = execute_buy(capital, row['close'], cost_rate)

            elif row['signal'] == 'sell' and position > 0:
                position, capital = execute_sell(position, row['close'], cost_rate)

        equity.append(position * row['close'] if position > 0 else capital)

    return equity, records


def bps_to_rate(cost_bps):
    return float(cost_bps) / 10000.0


def execute_buy(capital, price, cost_rate):
    """
    买入时扣除交易成本。
    """
    if capital <= 0 or price <= 0:
        return 0.0, capital

    position = capital * (1.0 - cost_rate) / price
    capital = 0.0
    return position, capital


def execute_sell(position, price, cost_rate):
    """
    卖出时扣除交易成本。
    """
    if position <= 0 or price <= 0:
        return position, 0.0

    capital = position * price * (1.0 - cost_rate)
    position = 0.0
    return position, capital


def apply_filter_and_trade(df, model, cfg, scaler, feature_cols,
                           signal_locs=None,
                           start_loc=None,
                           end_loc=None,
                           threshold=None,
                           cost_bps=0):
    """
    CTAML 过滤 buy。
    buy: p_good >= threshold 且空仓
    sell: 直接平仓
    """
    cost_rate = bps_to_rate(cost_bps)

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

            if row['signal'] == 'buy':
                x_win = scaler.transform(
                    df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values
                )
                x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)

                with torch.no_grad():
                    good_logit = model(x_ten)
                    p_good = torch.sigmoid(good_logit)[0].item()

                executed = (p_good >= threshold)

                signal_records.append({
                    'time': idx,
                    'price': row['close'],
                    'p_good': p_good,
                    'threshold': threshold,
                    'executed': executed
                })

                if executed and position == 0:
                    position, capital = execute_buy(capital, row['close'], cost_rate)

            elif row['signal'] == 'sell' and position > 0:
                position, capital = execute_sell(position, row['close'], cost_rate)

        equity.append(position * row['close'] if position > 0 else capital)

    return equity, signal_records


def backtest_no_filter(df, cfg, signal_locs=None, start_loc=None, end_loc=None, cost_bps=0):
    cost_rate = bps_to_rate(cost_bps)

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
                position, capital = execute_buy(capital, row['close'], cost_rate)

            elif row['signal'] == 'sell' and position > 0:
                position, capital = execute_sell(position, row['close'], cost_rate)

        equity.append(position * row['close'] if position > 0 else capital)

    return equity


def predict_for_indices(df, model, cfg, scaler, feature_cols, signal_indices):
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

        with torch.no_grad():
            good_logit = model(x_ten)
            p_good = torch.sigmoid(good_logit)[0].item()

        records.append({
            'loc': i_loc,
            'time': df.index[i_loc],
            'signal': row['signal'],
            'price': row['close'],
            'p_good': p_good
        })

    return pd.DataFrame(records)


def attach_labels_to_prediction(df, pred_df, cfg):
    """
    给预测结果附加通用标签列：
    - label_value
    - meta_label

    这样阈值校准时不再依赖 D_label，而是直接使用 meta_label。
    """
    if pred_df is None or len(pred_df) == 0:
        return pd.DataFrame()

    label_cols = [cfg.label_value_col, cfg.meta_label_col]
    missing_cols = [c for c in label_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"df 缺少标签列: {missing_cols}")

    label_df = df[label_cols].copy()
    label_df = label_df.reset_index().rename(columns={'index': 'time'})
    label_df['time'] = pd.to_datetime(label_df['time'])

    out = pred_df.copy()
    out['time'] = pd.to_datetime(out['time'])
    out = out.merge(label_df, on='time', how='left')

    return out


def calibrate_threshold_binary(y, p, cfg, name="model"):
    """
    通用二分类概率阈值校准。
    用于 CTAML 和 XGB。
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    if len(y) == 0 or len(p) == 0:
        print(f"{name}: 验证集为空，使用 fallback threshold")
        return cfg.fallback_pgood_threshold

    base_rate = y.mean()

    print(f"\n===== {name} 全局验证集阈值校准 =====")
    print(f"验证样本数: {len(y)}")
    print(f"good 数: {int(y.sum())}, good 占比: {base_rate:.2%}")
    print(f"p min/median/max: {p.min():.4f} / {np.median(p):.4f} / {p.max():.4f}")

    if len(np.unique(y)) < 2:
        q = 1.0 - cfg.desired_trade_ratio
        theta = float(np.quantile(p, q))
        print(f"{name}: 验证集类别单一，使用分位数阈值 {theta:.4f}")
        return theta

    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)

    print(f"{name}: Val AUC={auc:.4f}, AP={ap:.4f}, AP/base={ap / max(base_rate, 1e-12):.4f}")

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

        if cfg.threshold_method == 'global_f1_constrained':
            if cov < cfg.min_val_coverage or cov > cfg.max_val_coverage:
                continue

        valid_candidate_count += 1

        f1 = f1_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)

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

    print(f"{name}: threshold={best_theta:.4f}")
    print(
        f"{name}: precision={best_prec:.4f}, recall={best_rec:.4f}, "
        f"F1={best_f1:.4f}, coverage={best_cov:.2%}"
    )

    return best_theta


def calibrate_global_threshold_from_val(val_pred_all, cfg):
    """
    使用所有股票验证集上的 meta_label 和 p_good 校准全局阈值。
    """
    if val_pred_all is None or len(val_pred_all) == 0:
        print("全局验证集为空，使用 fallback 阈值")
        return cfg.fallback_pgood_threshold

    dfv = val_pred_all[
        (val_pred_all['signal'] == cfg.train_signal_type)
    ].dropna(subset=['p_good', cfg.meta_label_col]).copy()

    if len(dfv) == 0:
        print("全局验证集 buy 为空，使用 fallback 阈值")
        return cfg.fallback_pgood_threshold

    y = dfv[cfg.meta_label_col].values.astype(int)
    p = dfv['p_good'].values.astype(float)

    return calibrate_threshold_binary(y, p, cfg, name="MetaLabel-Model")


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


def backtest_buy_and_hold(df, start_loc=None, end_loc=None, cost_bps=0):
    """
    Buy and Hold 基线。
    为公平起见，使用 round-trip 成本，即买入和最终清算各扣一次成本。
    """
    cost_rate = bps_to_rate(cost_bps)

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

    # 每一天按“若当天清算”的净值计算
    equity = prices / first_price
    equity = equity * (1.0 - cost_rate) * (1.0 - cost_rate)
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
        equity_model,
        equity_signal,
        equity_buy_hold,
        equity_xgb=None
):
    """
    构造单股票逐日收益率曲线。

    当前通用框架只保留：
    - Meta-Label Model
    - Original Signal
    - Triple Barrier + XGB
    - Buy and Hold
    """
    start_loc = max(0, start_loc)
    end_loc = min(end_loc, len(df))

    if end_loc <= start_loc:
        return pd.DataFrame()

    dates = df.index[start_loc:end_loc]
    target_len = len(dates)

    eq_model = align_equity_to_dates(equity_model, target_len)
    eq_signal = align_equity_to_dates(equity_signal, target_len)
    eq_bh = align_equity_to_dates(equity_buy_hold, target_len)
    eq_xgb = align_equity_to_dates(equity_xgb, target_len) if equity_xgb is not None else np.full(target_len, np.nan)

    test_signal_locs = set(test_signal_locs) if test_signal_locs is not None else set()
    locs = list(range(start_loc, end_loc))

    curve_df = pd.DataFrame({
        'code': code,
        'time': dates,
        'close': df['close'].iloc[start_loc:end_loc].values,
        'raw_signal': df['signal'].iloc[start_loc:end_loc].values,
        'is_test_signal': [loc in test_signal_locs for loc in locs],

        'equity_model': eq_model,
        'return_model': eq_model - 1.0,

        'equity_signal': eq_signal,
        'return_signal': eq_signal - 1.0,

        'equity_xgb_tb': eq_xgb,
        'return_xgb_tb': eq_xgb - 1.0,

        'equity_buy_hold': eq_bh,
        'return_buy_hold': eq_bh - 1.0,
    })

    return curve_df


def plot_single_equity_curve(curve_df, cfg, code, cost_bps=0):
    """
    绘制单只股票累计收益率曲线。
    """
    if curve_df is None or len(curve_df) == 0:
        return None

    os.makedirs(cfg.plot_dir, exist_ok=True)

    dfp = curve_df.copy()
    dfp['time'] = pd.to_datetime(dfp['time'])

    plt.figure(figsize=(12, 6))

    plt.plot(dfp['time'], dfp['return_signal'], label='Original Signal', linewidth=1.8)
    plt.plot(dfp['time'], dfp['return_model'], label='Meta-Label Model', linewidth=1.8)
    plt.plot(dfp['time'], dfp['return_buy_hold'], label='Buy & Hold', linewidth=1.8)

    if 'return_xgb_tb' in dfp.columns and dfp['return_xgb_tb'].notna().any():
        plt.plot(dfp['time'], dfp['return_xgb_tb'], label='Triple Barrier + XGB', linewidth=1.4, linestyle=':')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'{cfg.run_tag} - {code} cumulative return, cost={cost_bps}bps')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        cfg.plot_dir,
        f"curve_{cfg.run_tag}_{code}_cost{cost_bps}bps.png"
    )
    plt.savefig(out_path, dpi=cfg.plot_dpi)
    plt.close()

    return out_path


def plot_average_equity_curve(all_curve_df, cfg, cost_bps=0):
    """
    按交易步数对齐，绘制所有股票平均累计收益率曲线。
    """
    if all_curve_df is None or len(all_curve_df) == 0:
        return None

    os.makedirs(cfg.plot_dir, exist_ok=True)

    dfs = []

    for code, g in all_curve_df.groupby('code'):
        g = g.sort_values('time').reset_index(drop=True).copy()
        g['step'] = np.arange(len(g))
        dfs.append(g)

    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    avg_df = df_all.groupby('step')[
        [
            'return_signal',
            'return_model',
            'return_xgb_tb',
            'return_buy_hold',
        ]
    ].mean(numeric_only=True).reset_index()

    plt.figure(figsize=(12, 6))

    plt.plot(avg_df['step'], avg_df['return_signal'], label='Original Signal', linewidth=2)
    plt.plot(avg_df['step'], avg_df['return_model'], label='Meta-Label Model', linewidth=2)
    plt.plot(avg_df['step'], avg_df['return_buy_hold'], label='Buy & Hold', linewidth=2)

    if avg_df['return_xgb_tb'].notna().any():
        plt.plot(avg_df['step'], avg_df['return_xgb_tb'], label='Triple Barrier + XGB', linestyle=':')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'{cfg.run_tag} average cumulative return, cost={cost_bps}bps')
    plt.xlabel('Aligned trading day')
    plt.ylabel('Average Cumulative Return')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        cfg.plot_dir,
        f"curve_{cfg.run_tag}_AVERAGE_cost{cost_bps}bps.png"
    )
    plt.savefig(out_path, dpi=cfg.plot_dpi)
    plt.close()

    return out_path


def bootstrap_mean_ci(x, n_boot=5000, alpha=0.05, seed=42):
    """
    bootstrap 均值置信区间。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    means = []
    n = len(x)

    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        means.append(np.mean(sample))

    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)
    mean = np.mean(x)

    return mean, lower, upper


def compute_significance_tests(summary_df, cfg, seed=42):
    """
    对每只股票的改善做 Wilcoxon 和 bootstrap CI。

    当前通用框架比较：
    - Meta-Label Model vs Original Signal
    - Triple Barrier + XGB vs Original Signal
    - Meta-Label Model vs Triple Barrier + XGB
    """
    tests = []

    comparisons = [
        ('model_vs_base_sharpe', 'delta_sharpe_model_vs_base'),
        ('xgb_vs_base_sharpe', 'delta_sharpe_xgb_vs_base'),

        ('model_vs_base_mdd', 'mdd_improve_model_vs_base'),
        ('xgb_vs_base_mdd', 'mdd_improve_xgb_vs_base'),

        ('model_vs_base_return', 'delta_return_model_vs_base'),
        ('xgb_vs_base_return', 'delta_return_xgb_vs_base'),

        ('model_vs_xgb_sharpe', 'delta_sharpe_model_vs_xgb'),
        ('model_vs_xgb_mdd', 'mdd_improve_model_vs_xgb'),
        ('model_vs_xgb_return', 'delta_return_model_vs_xgb'),
    ]

    for name, col in comparisons:
        if col not in summary_df.columns:
            continue

        x = summary_df[col].values.astype(float)
        x = x[np.isfinite(x)]

        if len(x) < 3:
            stat = np.nan
            pval = np.nan
        else:
            try:
                stat, pval = wilcoxon(x, alternative='two-sided')
            except Exception:
                stat, pval = np.nan, np.nan

        mean, ci_low, ci_high = bootstrap_mean_ci(
            x,
            n_boot=5000,
            alpha=0.05,
            seed=seed
        )

        tests.append({
            'comparison': name,
            'metric_col': col,
            'n': len(x),
            'mean': mean,
            'bootstrap_ci_low': ci_low,
            'bootstrap_ci_high': ci_high,
            'wilcoxon_stat': stat,
            'wilcoxon_pvalue': pval
        })

    test_df = pd.DataFrame(tests)

    out_path = os.path.join(
        cfg.results_dir,
        f"significance_{cfg.run_tag}_seed{seed}.csv"
    )
    test_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"显著性检验结果已保存: {out_path}")

    return test_df


def analyze_prediction_vs_label(df, buy_records, cfg):
    """
    通用预测结果诊断。

    不再假设标签叫 D_label，而是使用：
    - cfg.label_value_col
    - cfg.meta_label_col
    """
    if not buy_records:
        print("无 buy_records 可分析")
        return None

    rec = pd.DataFrame(buy_records).copy()
    rec['time'] = pd.to_datetime(rec['time'])

    label_df = df[[cfg.label_value_col, cfg.meta_label_col]].copy()
    label_df = label_df.reset_index().rename(columns={'index': 'time'})
    label_df['time'] = pd.to_datetime(label_df['time'])

    merged = rec.merge(label_df, on='time', how='left')

    print("\n===== 预测 vs 真实 meta_label 诊断 =====")
    print(f"buy 信号数: {len(merged)}")
    print(f"真实好信号数 meta_label=1: {int(merged[cfg.meta_label_col].sum(skipna=True))}")
    print(f"模型执行数: {int(merged['executed'].sum())}")

    if merged['executed'].sum() > 0:
        exec_df = merged[merged['executed']]
        print(f"执行信号中的真实好信号数: {exec_df[cfg.meta_label_col].sum()} / {len(exec_df)}")
        print(f"执行信号真实好信号占比: {exec_df[cfg.meta_label_col].mean():.2%}")
        print(f"执行信号平均 label_value: {exec_df[cfg.label_value_col].mean():.4f}")

    skip_df = merged[~merged['executed']]
    if len(skip_df) > 0:
        print(f"跳过信号中的真实好信号数: {skip_df[cfg.meta_label_col].sum()} / {len(skip_df)}")
        print(f"跳过信号真实好信号占比: {skip_df[cfg.meta_label_col].mean():.2%}")
        print(f"跳过信号平均 label_value: {skip_df[cfg.label_value_col].mean():.4f}")

    if merged[cfg.meta_label_col].notna().sum() > 3 and 'p_good' in merged.columns:
        corr_p = merged[['p_good', cfg.meta_label_col]].corr(method='spearman').iloc[0, 1]
        print(f"p_good 与 meta_label 的 Spearman 相关: {corr_p:.4f}，理论上应为正相关")

    cols = ['time', 'price', 'p_good', cfg.label_value_col, cfg.meta_label_col, 'executed']
    cols = [c for c in cols if c in merged.columns]

    print("\n按 p_good 排序：")
    print(merged[cols].sort_values('p_good', ascending=False).to_string(index=False))

    return merged


# ================== 8. 主程序（示例） ==================
def run_one_seed(seed):
    cfg.seed = seed
    set_seed(seed)
    # 把原 main() 中除了 set_seed(cfg.seed) 以外的全部内容放到这里
    # 最后返回 summary_df
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
            "NFLX", "AAPL", "MSFT", "TSLA", "WMT", "KO"
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

        # ================= 通用标注 API =================
        # 根据 cfg.label_method 创建标注器。
        # 标注器会在 df 中生成：
        # - cfg.label_value_col
        # - cfg.meta_label_col
        labeler = build_labeler(cfg.label_method)
        df, label_info = labeler.transform(df, cfg)

        print(f"股票 {code} 使用标注方法: {label_info.get('labeler', cfg.label_method)}")

        train_indices, val_indices, test_indices = split_signals_train_val_test(
            df, cfg, train_ratio=0.6, val_ratio=0.2
        )

        # 标签诊断
        print_label_diagnostics(
            code, df, train_indices,
            label_value_col=cfg.label_value_col,
            meta_label_col=cfg.meta_label_col,
            prefix="训练期"
        )
        print_label_diagnostics(
            code, df, val_indices,
            label_value_col=cfg.label_value_col,
            meta_label_col=cfg.meta_label_col,
            prefix="验证期"
        )
        print_label_diagnostics(
            code, df, test_indices,
            label_value_col=cfg.label_value_col,
            meta_label_col=cfg.meta_label_col,
            prefix="测试期"
        )
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
            code, df, feature_cols, label_info,
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

    for code, df, feature_cols, label_info, train_indices, val_indices, test_indices, _ in stock_data_info:
        train_ds = SignalDataset(
            df, feature_cols, cfg,
            scaler=global_scaler,
            signal_indices=train_indices,
            signal_filter_type=cfg.train_signal_type
        )

        val_ds = SignalDataset(
            df, feature_cols, cfg,
            scaler=global_scaler,
            signal_indices=val_indices,
            signal_filter_type=cfg.train_signal_type
        )

        test_ds = SignalDataset(
            df, feature_cols, cfg,
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
            label_info,
            train_indices, val_indices, test_indices
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
    model = MetaLabelSequenceClassifier(
        input_dim=len(feature_cols),
        d_model=cfg.d_model
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_model_path = os.path.join(
        cfg.results_dir,
        f"best_ctaml_{cfg.run_tag}_seed{cfg.seed}.pth"
    )

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
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("早停触发")
                break

    # 加载最佳模型用于测试
    model_eval = MetaLabelSequenceClassifier(
        input_dim=len(feature_cols),
        d_model=cfg.d_model
    ).to(cfg.device)

    model_eval.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
    model_eval.eval()
    # ---------- 全局验证集阈值校准 ----------
    global_val_preds = []

    for code, df, test_ds, scaler, feature_cols, label_info, train_indices, val_indices, test_indices in test_info:
        if len(val_indices) == 0:
            continue

        pred_val = predict_for_indices(
            df, model_eval, cfg, scaler, feature_cols, val_indices
        )

        pred_val = attach_labels_to_prediction(df, pred_val, cfg)

        if len(pred_val) > 0:
            pred_val['code'] = code
            global_val_preds.append(pred_val)

    if len(global_val_preds) > 0:
        global_val_pred_df = pd.concat(global_val_preds, axis=0, ignore_index=True)
    else:
        global_val_pred_df = pd.DataFrame()

    global_threshold = calibrate_global_threshold_from_val(global_val_pred_df, cfg)
    print(f"\n最终全局 p_good threshold = {global_threshold:.4f}")

    # ================= Triple Barrier + XGB 外部基线 =================
    xgb_model = None
    xgb_threshold = cfg.fallback_pgood_threshold

    if cfg.enable_triple_barrier_xgb:
        all_xgb_train_X = []
        all_xgb_train_y = []
        all_xgb_val_y = []
        all_xgb_val_p = []

        for item in test_info:
            code, df, test_ds, scaler, feature_cols, label_info, train_indices, val_indices, test_indices = item

            tb_train_labels = compute_triple_barrier_labels(df, train_indices, cfg)
            tb_val_labels = compute_triple_barrier_labels(df, val_indices, cfg)

            X_train, y_train, _ = build_xgb_dataset(df, feature_cols, train_indices, tb_train_labels, cfg)

            if len(y_train) > 0:
                all_xgb_train_X.append(X_train)
                all_xgb_train_y.append(y_train)

        if len(all_xgb_train_X) > 0:
            X_train_all = np.concatenate(all_xgb_train_X, axis=0)
            y_train_all = np.concatenate(all_xgb_train_y, axis=0)

            xgb_model = train_triple_barrier_xgb(X_train_all, y_train_all, seed=cfg.seed)

            # 验证集校准 XGB 阈值
            if xgb_model is not None:
                for item in test_info:
                    code, df, test_ds, scaler, feature_cols, label_info, train_indices, val_indices, test_indices = item

                    tb_val_labels = compute_triple_barrier_labels(df, val_indices, cfg)
                    X_val, y_val, _ = build_xgb_dataset(df, feature_cols, val_indices, tb_val_labels, cfg)

                    if len(y_val) > 0:
                        p_val = predict_xgb_proba(xgb_model, X_val)
                        all_xgb_val_y.extend(y_val.tolist())
                        all_xgb_val_p.extend(p_val.tolist())

                if len(all_xgb_val_y) > 0:
                    xgb_threshold = calibrate_threshold_binary(
                        all_xgb_val_y,
                        all_xgb_val_p,
                        cfg,
                        name="TripleBarrier-XGB"
                    )

        print(f"TripleBarrier-XGB threshold = {xgb_threshold:.4f}")

    # 对每只股票单独回测并汇总指标
    summary = []

    # 保存所有股票的逐日收益率曲线，最后合并输出
    all_equity_curve_dfs = []

    for code, df, test_ds, scaler, feature_cols, label_info, train_indices, val_indices, test_indices in test_info:
        if len(test_indices) == 0:
            print(f"股票 {code} 测试信号为空，跳过")
            continue

        test_start_loc = max(min(test_indices), cfg.window_size)
        test_end_loc = len(df)
        threshold = global_threshold

        for cost_bps in cfg.cost_bps_list:
            # ---------- 1. Meta-Label Model ----------
            equity_model, buy_records = apply_filter_and_trade(
                df, model_eval, cfg, scaler, feature_cols,
                signal_locs=test_indices,
                start_loc=test_start_loc,
                end_loc=test_end_loc,
                threshold=threshold,
                cost_bps=cost_bps
            )

            # ---------- 2. Original Signal ----------
            equity_b = backtest_no_filter(
                df, cfg,
                signal_locs=test_indices,
                start_loc=test_start_loc,
                end_loc=test_end_loc,
                cost_bps=cost_bps
            )

            # ---------- 3. Triple Barrier + XGB ----------
            if cfg.enable_triple_barrier_xgb and xgb_model is not None:
                equity_xgb, xgb_records = apply_xgb_filter_and_trade(
                    df, xgb_model, cfg, feature_cols,
                    signal_locs=test_indices,
                    start_loc=test_start_loc,
                    end_loc=test_end_loc,
                    threshold=xgb_threshold,
                    cost_bps=cost_bps
                )
            else:
                equity_xgb, xgb_records = [1.0], []

            # ---------- 4. Buy and Hold ----------
            equity_h = backtest_buy_and_hold(
                df,
                start_loc=test_start_loc,
                end_loc=test_end_loc,
                cost_bps=cost_bps
            )

            # ---------- 7. 指标 ----------
            sharpe_model = calc_sharpe(equity_model)
            mdd_model = calc_max_drawdown(equity_model)
            ret_model = calc_total_return(equity_model)

            sharpe_b = calc_sharpe(equity_b)
            mdd_b = calc_max_drawdown(equity_b)
            ret_b = calc_total_return(equity_b)

            sharpe_xgb = calc_sharpe(equity_xgb)
            mdd_xgb = calc_max_drawdown(equity_xgb)
            ret_xgb = calc_total_return(equity_xgb)

            sharpe_h = calc_sharpe(equity_h)
            mdd_h = calc_max_drawdown(equity_h)
            ret_h = calc_total_return(equity_h)

            # ---------- 8. 诊断打印 ----------
            if buy_records:
                df_rec = pd.DataFrame(buy_records)

                print(f"\n===== 股票 {code} 测试集买入信号 p_good 分析 | cost={cost_bps}bps =====")
                print(f"测试集买入信号数: {len(df_rec)}")
                print(f"threshold: {threshold:.4f}")
                print(f"p_good 最小值: {df_rec['p_good'].min():.4f}")
                print(f"p_good 最大值: {df_rec['p_good'].max():.4f}")
                print(f"p_good 中位数: {df_rec['p_good'].median():.4f}")

                executed_count = int(df_rec['executed'].sum())
                print(f"实际执行买入信号数: {executed_count}")

                if executed_count > 0:
                    print("执行的买入信号详情:")
                    print(
                        df_rec[df_rec['executed']][['time', 'price', 'p_good', 'threshold']]
                        .to_string(index=False)
                    )
            else:
                print(f"股票 {code} 测试集无买入信号记录 | cost={cost_bps}bps")

            # ---------- 9. 收益曲线 ----------
            curve_df = build_equity_curve_df(
                code=code,
                df=df,
                start_loc=test_start_loc,
                end_loc=test_end_loc,
                test_signal_locs=test_indices,
                equity_model=equity_model,
                equity_signal=equity_b,
                equity_buy_hold=equity_h,
                equity_xgb=equity_xgb
            )

            if len(curve_df) > 0:
                curve_df['seed'] = cfg.seed
                curve_df['cost_bps'] = cost_bps

            if cfg.save_outputs and cfg.save_equity_curves and len(curve_df) > 0:
                curve_dir = os.path.join(cfg.results_dir, "equity_curves")
                os.makedirs(curve_dir, exist_ok=True)

                curve_path = os.path.join(
                    curve_dir,
                    f"equity_{cfg.run_tag}_{code}_seed{cfg.seed}_cost{cost_bps}bps.csv"
                )

                curve_df.to_csv(curve_path, index=False, encoding='utf-8-sig')
                print(f"收益率曲线已保存: {curve_path}")

                if cfg.save_plots and cost_bps == 0:
                    fig_path = plot_single_equity_curve(curve_df, cfg, code, cost_bps=cost_bps)
                    print(f"收益曲线图已保存: {fig_path}")

            if len(curve_df) > 0:
                all_equity_curve_dfs.append(curve_df)

            # ---------- 10. 覆盖率 ----------
            buy_count = len(buy_records)
            exec_count = int(pd.DataFrame(buy_records)['executed'].sum()) if buy_records else 0
            coverage = exec_count / buy_count if buy_count > 0 else 0.0

            xgb_buy_count = len(xgb_records)
            xgb_exec_count = int(pd.DataFrame(xgb_records)['executed'].sum()) if xgb_records else 0

            summary.append((
                cfg.seed,
                cost_bps,
                code,

                sharpe_model, mdd_model, ret_model,
                sharpe_b, mdd_b, ret_b,
                sharpe_xgb, mdd_xgb, ret_xgb,
                sharpe_h, mdd_h, ret_h,

                buy_count, exec_count, coverage,
                xgb_buy_count, xgb_exec_count
            ))

            print(
                f"\n股票 {code} | cost={cost_bps}bps: "
                f"Model 夏普={sharpe_model:.4f}, MDD={mdd_model:.4%}, 收益={ret_model:.4%} | "
                f"Original 夏普={sharpe_b:.4f}, MDD={mdd_b:.4%}, 收益={ret_b:.4%} | "
                f"XGB 夏普={sharpe_xgb:.4f}, MDD={mdd_xgb:.4%}, 收益={ret_xgb:.4%} | "
                f"Buy&Hold 夏普={sharpe_h:.4f}, MDD={mdd_h:.4%}, 收益={ret_h:.4%}"
            )

            # 避免交易成本敏感性时重复打印过多明细，只在 cost=0 时分析一次
            if cost_bps == 0:
                analyze_prediction_vs_label(df, buy_records, cfg)

    if summary:
        if cfg.save_outputs:
            summary_df = pd.DataFrame(
                summary,
                columns=[
                    'seed',
                    'cost_bps',
                    'code',

                    'sharpe_model', 'mdd_model', 'return_model',
                    'sharpe_baseline', 'mdd_baseline', 'return_baseline',
                    'sharpe_xgb', 'mdd_xgb', 'return_xgb',
                    'sharpe_buy_hold', 'mdd_buy_hold', 'return_buy_hold',

                    'buy_count', 'exec_count', 'coverage',
                    'xgb_buy_count', 'xgb_exec_count'
                ]
            )

            summary_df['delta_sharpe_model_vs_base'] = (
                    summary_df['sharpe_model'] - summary_df['sharpe_baseline']
            )
            summary_df['delta_sharpe_xgb_vs_base'] = (
                    summary_df['sharpe_xgb'] - summary_df['sharpe_baseline']
            )
            summary_df['delta_sharpe_model_vs_xgb'] = (
                    summary_df['sharpe_model'] - summary_df['sharpe_xgb']
            )

            summary_df['mdd_improve_model_vs_base'] = (
                    summary_df['mdd_model'] - summary_df['mdd_baseline']
            )
            summary_df['mdd_improve_xgb_vs_base'] = (
                    summary_df['mdd_xgb'] - summary_df['mdd_baseline']
            )
            summary_df['mdd_improve_model_vs_xgb'] = (
                    summary_df['mdd_model'] - summary_df['mdd_xgb']
            )

            summary_df['delta_return_model_vs_base'] = (
                    summary_df['return_model'] - summary_df['return_baseline']
            )
            summary_df['delta_return_xgb_vs_base'] = (
                    summary_df['return_xgb'] - summary_df['return_baseline']
            )
            summary_df['delta_return_model_vs_xgb'] = (
                    summary_df['return_model'] - summary_df['return_xgb']
            )

            summary_df['delta_return_model_vs_buy_hold'] = (
                    summary_df['return_model'] - summary_df['return_buy_hold']
            )
            summary_df['delta_return_base_vs_buy_hold'] = (
                    summary_df['return_baseline'] - summary_df['return_buy_hold']
            )

            out_path = os.path.join(
                cfg.results_dir,
                f"summary_{cfg.run_tag}_seed{cfg.seed}.csv"
            )
            summary_df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"summary 已保存: {out_path}")

            print("\n===== 汇总结果：按交易成本分组 =====")

            main_group = summary_df.groupby('cost_bps').agg(
                avg_sharpe_model=('sharpe_model', 'mean'),
                avg_mdd_model=('mdd_model', 'mean'),
                avg_return_model=('return_model', 'mean'),

                avg_sharpe_baseline=('sharpe_baseline', 'mean'),
                avg_mdd_baseline=('mdd_baseline', 'mean'),
                avg_return_baseline=('return_baseline', 'mean'),

                avg_sharpe_xgb=('sharpe_xgb', 'mean'),
                avg_mdd_xgb=('mdd_xgb', 'mean'),
                avg_return_xgb=('return_xgb', 'mean'),

                avg_sharpe_buy_hold=('sharpe_buy_hold', 'mean'),
                avg_mdd_buy_hold=('mdd_buy_hold', 'mean'),
                avg_return_buy_hold=('return_buy_hold', 'mean'),

                total_buy=('buy_count', 'sum'),
                total_exec=('exec_count', 'sum'),
                avg_coverage=('coverage', 'mean')
            ).reset_index()

            main_group['global_coverage'] = main_group['total_exec'] / main_group['total_buy'].replace(0, np.nan)

            print(main_group.to_string(index=False))

            cost_summary = summary_df.groupby('cost_bps').agg({
                'sharpe_model': ['mean', 'std'],
                'mdd_model': ['mean', 'std'],
                'return_model': ['mean', 'std'],

                'sharpe_baseline': ['mean', 'std'],
                'mdd_baseline': ['mean', 'std'],
                'return_baseline': ['mean', 'std'],

                'sharpe_xgb': ['mean', 'std'],
                'mdd_xgb': ['mean', 'std'],
                'return_xgb': ['mean', 'std'],

                'sharpe_buy_hold': ['mean', 'std'],
                'mdd_buy_hold': ['mean', 'std'],
                'return_buy_hold': ['mean', 'std'],
            }).reset_index()

            cost_summary.columns = [
                '_'.join(col).strip('_') if isinstance(col, tuple) else col
                for col in cost_summary.columns
            ]

            cost_path = os.path.join(
                cfg.results_dir,
                f"cost_sensitivity_{cfg.run_tag}_seed{cfg.seed}.csv"
            )
            cost_summary.to_csv(cost_path, index=False, encoding='utf-8-sig')
            print(f"交易成本敏感性结果已保存: {cost_path}")

            # 只对 cost=0 的主结果做显著性检验
            summary_cost0 = summary_df[summary_df['cost_bps'] == 0].copy()

            if len(summary_cost0) > 0:
                sig_df = compute_significance_tests(summary_cost0, cfg, seed=cfg.seed)
                print(sig_df.to_string(index=False))

            print("\n===== 改善统计 =====")
            print(
                f"Meta-Label Model 胜过无过滤的股票数: "
                f"{(summary_df['delta_sharpe_model_vs_base'] > 0).sum()} / {len(summary_df)}"
            )
            print(
                f"Meta-Label Model 平均夏普提升: {summary_df['delta_sharpe_model_vs_base'].mean():.4f}"
            )
            print(
                f"Meta-Label Model 平均MDD改善: {summary_df['mdd_improve_model_vs_base'].mean():.4%}"
            )
            print(
                f"Meta-Label Model 平均最终收益提升: {summary_df['delta_return_model_vs_base'].mean():.4%}"
            )
            print(
                f"Meta-Label Model 胜过Buy&Hold的股票数: "
                f"{(summary_df['delta_return_model_vs_buy_hold'] > 0).sum()} / {len(summary_df)}"
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
                    f"equity_{cfg.run_tag}_seed{cfg.seed}_ALL.csv"
                )
                all_curve_df.to_csv(all_curve_path, index=False, encoding='utf-8-sig')
                print(f"全部股票收益率曲线已保存: {all_curve_path}")

                if cfg.save_plots:
                    for cost_bps in cfg.cost_bps_list:
                        sub_curve = all_curve_df[all_curve_df['cost_bps'] == cost_bps].copy()
                        if len(sub_curve) > 0:
                            avg_fig_path = plot_average_equity_curve(sub_curve, cfg, cost_bps=cost_bps)
                            print(f"平均收益曲线图已保存: {avg_fig_path}")

            return summary_df


def main():
    if cfg.save_outputs:
        os.makedirs(cfg.results_dir, exist_ok=True)

    if cfg.run_multi_seed:
        all_seed_summaries = []

        for seed in cfg.seeds:
            print(f"\n\n==============================")
            print(f"开始运行 seed = {seed}")
            print(f"==============================\n")

            cfg.seed = seed
            summary_df = run_one_seed(seed)

            if summary_df is not None and len(summary_df) > 0:
                all_seed_summaries.append(summary_df)

        if len(all_seed_summaries) > 0:
            all_df = pd.concat(all_seed_summaries, axis=0, ignore_index=True)

            all_path = os.path.join(
                cfg.results_dir,
                f"summary_{cfg.run_tag}_ALL_SEEDS.csv"
            )
            all_df.to_csv(all_path, index=False, encoding='utf-8-sig')
            print(f"多随机种子总结果已保存: {all_path}")

            multi_seed_summary = all_df.groupby('cost_bps').agg({
                'sharpe_model': ['mean', 'std'],
                'mdd_model': ['mean', 'std'],
                'return_model': ['mean', 'std'],

                'sharpe_baseline': ['mean', 'std'],
                'mdd_baseline': ['mean', 'std'],
                'return_baseline': ['mean', 'std'],

                'sharpe_xgb': ['mean', 'std'],
                'mdd_xgb': ['mean', 'std'],
                'return_xgb': ['mean', 'std'],

                'sharpe_buy_hold': ['mean', 'std'],
                'mdd_buy_hold': ['mean', 'std'],
                'return_buy_hold': ['mean', 'std'],
            }).reset_index()

            multi_seed_summary.columns = [
                '_'.join(col).strip('_') if isinstance(col, tuple) else col
                for col in multi_seed_summary.columns
            ]

            ms_path = os.path.join(
                cfg.results_dir,
                f"multi_seed_summary_{cfg.run_tag}.csv"
            )
            multi_seed_summary.to_csv(ms_path, index=False, encoding='utf-8-sig')
            print(f"多随机种子均值方差结果已保存: {ms_path}")

    else:
        run_one_seed(cfg.seed)


if __name__ == "__main__":
    main()
