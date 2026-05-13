"""
对比尾部感知元标签 (CTAML)：完整实验代码
根据论文"Contrastive Tail-Aware Meta-Labeling: Self-Supervised Extreme Region Distance Field Learning for Trading Signal Filtering"
目标期刊：ESWA / EAAI / ASOC / Neurocomputing / KBS / Information Sciences

依赖：pandas, numpy, torch, scipy, sklearn, matplotlib, tqdm
"""
# 导入必要的科学计算、数据处理和深度学习库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import genpareto  # 用于广义帕累托分布拟合
from scipy.signal import find_peaks  # 用于极值点检测（备用）
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')  # 忽略警告，保持输出整洁


# ================== 配置 ==================
class Config:
    """
        全局配置类，集中管理所有超参数和路径。
        这种集中式配置方便消融实验（通过修改属性快速切换模型变体）和参数调优。
    """
    # 数据
    signal_file = "signals.csv"  # time,price,signal 时间,价格,信号（买入/卖出/none）
    ohlcv_file = "ohlcv.csv"  # time,open,high,low,close,volume 时间,开,高,低,收,量
    # 窗口与距离参数（对应论文公式1）
    window_size = 160  # 输入窗口长度（步数）
    future_T = 60  # 寻找极值点的未来窗口
    kappa = 1.5  # 距离归一化中的ATR乘数，控制价格维度的尺度
    # EVT相关
    evt_rolling = 252  # 用于GPD拟合的滚动窗口长度（对应约一年交易日）
    evt_quantile = 0.90  # 极端事件的收益率绝对值分位数阈值
    extrema_win = 3  # 极值点邻域半径（日）
    # 对比学习
    contrastive_win = 5  # 对比学习正样本窗口半径  正/负样本的环境窗口半径（论文取±5步）
    D_buckets = 8  # 样本均衡分桶数（此处预留，用于按D值分层采样）
    D_threshold = 0.4  # 固定阈值，若 use_quantile_threshold=False 则使用它

    # 是否用训练期预测 d_mid 的分位数自动校准阈值
    # True：例如 threshold_quantile=0.3 表示执行训练期预测质量最好的 30% 买入信号
    # False：使用固定 D_threshold
    use_quantile_threshold = True
    threshold_quantile = 0.30

    # Oracle 诊断回测阈值：直接用真实 D_label 过滤，仅用于诊断标签是否有效，不是实盘结果
    oracle_D_threshold = 0.4

    # 模型
    encoder_type = 'BiLSTM'  # 'BiLSTM' 或 'Informer'
    d_model = 128  # 隐藏层/嵌入维度
    n_heads = 8  # 多头注意力的头数
    proj_dim = 128  # 对比学习投影头输出维度
    quantiles = [0.1, 0.5, 0.9]  # 分位数回归的分位点
    use_taa = False  # 是否启用尾部感知注意力模块
    lam_init = 0.1  # TAA中可学习缩放参数λ的初始值
    tau_cl = 0.1  # InfoNCE损失的温度系数
    # 多任务损失权重
    lambda_D = 1.0
    lambda_xi = 0.5
    lambda_cl = 0.3

    # 训练
    batch_size = 64
    lr = 1e-3
    epochs = 200
    patience = 15  # 早停耐心计数器
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 回测（止损参数）
    stop_loss_c = 1.0  # 止损宽度系数
    stop_loss_alpha = 0.05  # VaR 分位数（用于 ξ 驱动止损）


cfg = Config()


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
    三元组数据集：
    - 信号窗口 x_win
    - D_label
    - xi_label
    - abs_ret
    - 正样本窗口 pos_win：对应方向的未来极值区域
    - 负样本窗口 neg_win
    """

    def __init__(self, df, feature_cols, cfg, extrema_info, scaler=None, signal_indices=None):
        self.df = df
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels_D = df['D_label'].values
        self.labels_xi = df['xi_label'].values
        self.window = cfg.window_size
        self.cfg = cfg

        # 兼容旧变量名：现在 extrema_info 是 dict: idx -> 'peak'/'valley'
        if isinstance(extrema_info, dict):
            self.extrema_info = extrema_info
        else:
            # 如果外部误传 set，则退化为空方向，不建议
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

        self.signal_indices = [
            i for i in raw_indices
            if i >= self.window
            and i + cfg.future_T + cfg.contrastive_win < len(df)
            and not np.isnan(self.labels_D[i])
            and not np.isnan(self.labels_xi[i])
        ]

        self.signal_types = [self.df['signal'].iloc[i] for i in self.signal_indices]

    def __len__(self):
        return len(self.signal_indices)

    def __getitem__(self, idx):
        i = self.signal_indices[idx]

        x_win = self.features[i - self.window:i]
        d_label = self.labels_D[i]
        xi_label = self.labels_xi[i]
        abs_ret = self.df['abs_returns'].values[i - self.window:i]

        pos_len = 2 * self.cfg.contrastive_win + 1
        pos_win = np.zeros((pos_len, self.features.shape[1]), dtype=np.float32)

        future_end = min(len(self.df), i + self.cfg.future_T + 1)
        sig_type = self.signal_types[idx]
        target_type = target_extrema_type(sig_type)

        # 正样本：买入找 valley，卖出找 peak
        if target_type is not None:
            for j in range(i, future_end):
                if j in self.extrema_info and self.extrema_info[j] == target_type:
                    start = j - self.cfg.contrastive_win
                    end = j + self.cfg.contrastive_win + 1

                    if start >= 0 and end <= len(self.df):
                        pos_win = self.features[start:end]
                    break

        # 负样本：远离所有极值点的普通区域
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
            torch.tensor(abs_ret, dtype=torch.float32), \
            torch.tensor(pos_win, dtype=torch.float32), \
            torch.tensor(neg_win, dtype=torch.float32)


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
            None,   # 不使用 ConvLayer，避免序列长度变化 [ConvLayer(d_model) for _ in range(e_layers-1)] if e_layers>1 else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        enc_out = self.embedding(x, None)   # (B, T, d_model)
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
        self.head_xi = nn.Linear(d_model, 1)  # 输出 ξ̂
        self.proj_head = ProjHead(d_model, proj_dim)

    def forward(self, x, abs_returns, xi_hat, sigma_hat):
        h_seq = self.encoder(x)                      # (B, T, d_model)
        if self.use_taa and abs_returns is not None:
            h_seq = self.taa(h_seq, abs_returns, xi_hat, sigma_hat)
        h = h_seq[:, -1, :]                          # 取最后时间步
        D_hat = self.head_D(h)
        xi_out = self.head_xi(h).squeeze(-1)
        proj = self.proj_head(h)
        return D_hat, xi_out, proj


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


# ================== 6. 训练循环 ==================
def train_epoch(model, dataloader, optimizer, cfg):
    model.train()
    total_loss = 0
    for x, d_label, xi_label, abs_ret, pos_win, _ in dataloader:
        x = x.to(cfg.device)
        d_label = d_label.to(cfg.device)
        xi_label = xi_label.to(cfg.device)
        abs_ret = abs_ret.to(cfg.device)
        pos_win = pos_win.to(cfg.device)

        # TAA 所需的 xi_hat 和 sigma_hat 使用当前窗口标签和标准差
        sigma_hat = torch.std(abs_ret, dim=1) + 1e-6
        D_hat, xi_out, proj_anchor = model(x, abs_ret, xi_label, sigma_hat)

        loss_D = pinball_loss(D_hat, d_label, cfg.quantiles)
        loss_xi = F.smooth_l1_loss(xi_out, xi_label)

        # 对比损失：正样本通过相同编码器获取投影
        h_pos_seq = model.encoder(pos_win)      # 序列
        h_pos = h_pos_seq[:, -1, :]
        proj_pos = model.proj_head(h_pos)

        # 过滤掉正样本为全零的无效样本（未找到同向极值点）
        pos_valid_mask = pos_win.reshape(pos_win.size(0), -1).abs().sum(dim=1) > 1e-8
        if pos_valid_mask.sum() > 1:
            proj_anchor_valid = proj_anchor[pos_valid_mask]
            proj_pos_valid = proj_pos[pos_valid_mask]
            loss_cl = info_nce(proj_anchor_valid, proj_pos_valid, cfg.tau_cl)
        else:
            loss_cl = torch.tensor(0.0, device=pos_win.device)

        loss = cfg.lambda_D * loss_D + cfg.lambda_xi * loss_xi + cfg.lambda_cl * loss_cl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def split_signals_by_time(df, cfg, train_ratio=0.7):
    """
    返回 train_indices, test_indices （在df中的位置索引）
    """
    signal_df = df[df['signal'] != 'none'].copy()
    if signal_df.empty:
        return [], []

    # 按时间排序
    signal_df = signal_df.sort_index()
    n = len(signal_df)
    split_idx = int(n * train_ratio)
    train_times = signal_df.index[:split_idx]
    test_times = signal_df.index[split_idx:]

    # 找出训练/测试信号在df中的绝对位置
    train_locs = [df.index.get_loc(t) for t in train_times]
    test_locs = [df.index.get_loc(t) for t in test_times]

    # 防止训练信号的未来窗口触及测试集时间
    # 取测试集的最小位置，要求 train_loc + cfg.future_T + cfg.contrastive_win < min_test_loc
    if test_locs:
        min_test_loc = min(test_locs)
        train_locs = [loc for loc in train_locs
                      if loc + cfg.future_T + cfg.contrastive_win < min_test_loc]
    return train_locs, test_locs


# ================== 7. 回测与决策 ==================
def apply_filter_and_trade(df, model, cfg, scaler, feature_cols,
                           signal_locs=None,
                           start_loc=None,
                           end_loc=None,
                           threshold=None):
    """
    过滤后回测。

    重要修正：
    1. 只在 signal_locs 指定的测试集信号上交易；
    2. buy 只有 position == 0 时才执行，避免重复 buy 把 position 清零；
    3. threshold 可传入校准阈值；
    4. executed 使用 <=，避免分位数阈值等于某些预测值时一个都不执行。
    """
    if threshold is None:
        threshold = cfg.D_threshold

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
            x_win = scaler.transform(df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values)
            x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)

            abs_ret = torch.tensor(
                df['abs_returns'].iloc[i_loc - cfg.window_size:i_loc].values,
                dtype=torch.float32
            ).unsqueeze(0).to(cfg.device)

            with torch.no_grad():
                D_hat, _, _ = model(
                    x_ten,
                    abs_ret,
                    torch.zeros(1).to(cfg.device),
                    torch.tensor([1.0]).to(cfg.device)
                )
                d_mid = D_hat[0, 1].item()

            if row['signal'] == 'buy':
                executed = (d_mid <= threshold)

                signal_records.append({
                    'time': idx,
                    'price': row['close'],
                    'd_mid': d_mid,
                    'threshold': threshold,
                    'executed': executed
                })

                # 关键修正：只有空仓时才能买入
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

def predict_dmid_for_indices(df, model, cfg, scaler, feature_cols, signal_indices):
    """
    对指定信号位置预测 d_mid。
    主要用于训练期阈值校准和测试期诊断。
    """
    records = []
    model.eval()

    for i_loc in signal_indices:
        if i_loc < cfg.window_size:
            continue

        row = df.iloc[i_loc]

        if row['signal'] == 'none':
            continue

        x_win = scaler.transform(df[feature_cols].iloc[i_loc - cfg.window_size:i_loc].values)
        x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)

        abs_ret = torch.tensor(
            df['abs_returns'].iloc[i_loc - cfg.window_size:i_loc].values,
            dtype=torch.float32
        ).unsqueeze(0).to(cfg.device)

        with torch.no_grad():
            D_hat, _, _ = model(
                x_ten,
                abs_ret,
                torch.zeros(1).to(cfg.device),
                torch.tensor([1.0]).to(cfg.device)
            )
            d_mid = D_hat[0, 1].item()

        records.append({
            'loc': i_loc,
            'time': df.index[i_loc],
            'signal': row['signal'],
            'price': row['close'],
            'd_mid': d_mid
        })

    return pd.DataFrame(records)


def calibrate_threshold_from_train(df, model, cfg, scaler, feature_cols, train_indices):
    """
    用训练期 buy 信号预测 d_mid 的分位数校准阈值。
    """
    if not cfg.use_quantile_threshold:
        return cfg.D_threshold

    pred_df = predict_dmid_for_indices(
        df, model, cfg, scaler, feature_cols, train_indices
    )

    if len(pred_df) == 0:
        return cfg.D_threshold

    buy_pred = pred_df[pred_df['signal'] == 'buy'].copy()

    if len(buy_pred) == 0:
        return cfg.D_threshold

    theta = float(np.quantile(buy_pred['d_mid'].values, cfg.threshold_quantile))

    # 加一个极小量，避免大量预测值相等时 <= 仍然执行不到
    theta = min(1.0, theta + 1e-6)

    print("\n===== 阈值校准 =====")
    print(f"训练期 buy 信号数: {len(buy_pred)}")
    print(f"d_mid min/median/max: {buy_pred['d_mid'].min():.4f} / {buy_pred['d_mid'].median():.4f} / {buy_pred['d_mid'].max():.4f}")
    print(f"使用分位数 q={cfg.threshold_quantile:.2f} 校准 threshold={theta:.4f}")

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


def analyze_prediction_vs_label(df, buy_records, good_threshold=0.4):
    """
    分析 CTAML 预测 d_mid 和真实 D_label 的对应关系。
    """
    if not buy_records:
        print("无 buy_records 可分析")
        return

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
        corr = merged[['d_mid', 'D_label']].corr(method='spearman').iloc[0, 1]
        print(f"d_mid 与 D_label 的 Spearman 相关: {corr:.4f}")

    print("\n按 d_mid 从小到大排序：")
    print(
        merged[['time', 'price', 'd_mid', 'D_label', 'true_good', 'executed']]
        .sort_values('d_mid')
        .to_string(index=False)
    )

    return merged


def labeled_signals():
    print("加载数据...")
    stocks_df = pd.read_csv("D://github//RobotMeQ//QuantData//asset_code//a800_stocks_2025.csv", dtype={'code': str})
    for _, row in stocks_df.iterrows():
        code = row['code']  # 如 'sh.600000'
        # 构造文件路径，注意你的命名规则
        code_short = code.split('.')[-1]  # '600000'
        ohlcv_path = f"D:/github/RobotMeQ_Dataset/QuantData/backTest/bar_A_{code_short}_d.csv"
        signal_path = f"D:/github/RobotMeQ_Dataset/QuantData/trade_point_backtest_c4_oscillation_boll_nature/A_{code_short}_d.csv"

        df, feature_cols = load_and_prepare(signal_path, ohlcv_path)
        print(code+"计算极值标注...")
        df, xi_series, sigma_series, extrema_info = compute_evt_distance_xi(df, cfg)

        # df.to_csv(code+"_labeled_data.csv")  # 保存标注结果

        # 将 D_label 和 xi_label 写回 signals.csv 对应的行
        signals_orig = pd.read_csv(signal_path, parse_dates=['time'])
        # 从标注结果中取出信号点的标签
        labeled_signals = df[df['signal'] != 'none'][['D_label', 'xi_label']].copy()
        labeled_signals.index.name = 'time'
        labeled_signals = labeled_signals.reset_index()
        # 按时间合并
        signals_labeled = signals_orig.merge(labeled_signals, on='time', how='left')
        signals_labeled.to_csv("./trade_point_backtest_c4_oscillation_boll_nature/"+code_short+"signals_labeled.csv", index=False)


# ================== 8. 主程序（示例） ==================
def main():
    # 股票列表（按你自己的路径修改）
    # stock_codes = ['AES', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']  # 可扩展，例如 ['AES', '600018', '600176']  'AES', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'
    # stock_codes = [
    #     "", "002311", "002493", "600588", "002049", "000977", "000617", "601111",
    #     "601006", "002475", "000876", "601600", "002179", "601818", "601117", "600438",
    #     "000568", "002304", "002050", "601318", "600111", "600426", "601618", "600176",
    #     "600460", "600415", "600019", "600893", "600104", "600031", "600089", "600115",
    #     "000661", "600233"
    # ]
    # 000977
    stock_codes = [
        "002179", "002311", "002493", "600588", "002049", "000977"
    ]
    # stocks_df = pd.read_csv("D://github//RobotMeQ//QuantData//asset_code//a800_stocks_2025.csv", dtype={'code': str})
    # for _, row in stocks_df.iterrows():
    #     code = row['code']  # 如 'sh.600000'
    #     stock_codes.append(code)
    #
    #     # 构造文件路径，注意你的命名规则
    #     code_short = code.split('.')[-1]  # '600000'
    ohlcv_paths = [f"D:/github/RobotMeQ_Dataset/QuantData/backTest/bar_A_{code}_d.csv" for code in stock_codes]
    signal_paths = [f"D:/github/RobotMeQ_Dataset/QuantData/trade_point_backtest_c4_oscillation_boll_nature/A_{code}_d.csv" for code in stock_codes]

    # ---------- 第一阶段：收集所有股票的训练集特征，用于拟合全局 scaler ----------
    global_train_features = []          # 存储每只股票训练期的特征数组
    stock_data_info = []                # 存储每只股票的处理信息，供第二阶段使用

    for code, sig_path, ohlcv_path in zip(stock_codes, signal_paths, ohlcv_paths):
        print(f"处理股票: {code}")
        cfg.signal_file = sig_path
        cfg.ohlcv_file = ohlcv_path

        # 加载并标注（标注使用全量数据）
        df, feature_cols = load_and_prepare(sig_path, ohlcv_path)
        df, xi_series, sigma_series, extrema_info = compute_evt_distance_xi(df, cfg)

        # 划分训练/测试信号索引
        train_indices, test_indices = split_signals_by_time(df, cfg, train_ratio=0.7)

        # 标签诊断
        print_label_diagnostics(code, df, train_indices, prefix="训练期")
        print_label_diagnostics(code, df, test_indices, prefix="测试期")
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"股票 {code} 训练或测试信号不足，跳过")
            continue

        # 确定训练期的时间范围（最后一个训练信号的时间）
        train_times = [df.index[idx] for idx in train_indices]  # 训练信号对应的时间戳
        end_train_time = max(train_times)                       # 训练期结束时间

        # 提取该股票训练期的所有特征行（用于 fit 全局 scaler）
        train_mask = df.index <= end_train_time
        train_features_stock = df[feature_cols].loc[train_mask].values
        global_train_features.append(train_features_stock)

        # 保存信息供第二阶段使用
        stock_data_info.append((
            code, df, feature_cols, extrema_info,
            train_indices, test_indices, end_train_time
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
    test_info = []   # 存储 (code, df, test_ds, scaler, feature_cols, extrema_idx_set)

    for code, df, feature_cols, extrema_info, train_indices, test_indices, _ in stock_data_info:
        train_ds = SignalDataset(
            df, feature_cols, cfg, extrema_info,
            scaler=global_scaler,
            signal_indices=train_indices
        )

        test_ds = SignalDataset(
            df, feature_cols, cfg, extrema_info,
            scaler=global_scaler,
            signal_indices=test_indices
        )

        if len(train_ds) == 0:
            print(f"股票 {code} 有 train_indices，但构造 train_ds 后样本数为 0，跳过")
            continue

        all_train_datasets.append(train_ds)

        # 这里额外保存 train_indices 和 test_indices，方便阈值校准和严格测试集回测
        test_info.append((
            code, df, test_ds, global_scaler, feature_cols,
            extrema_info, train_indices, test_indices
        ))

    if len(all_train_datasets) == 0:
        print("所有股票构造后的训练 Dataset 都为空，终止")
        return
    # 合并所有股票训练数据
    combined_train_ds = torch.utils.data.ConcatDataset(all_train_datasets)
    train_loader = DataLoader(combined_train_ds, batch_size=cfg.batch_size, shuffle=True)

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
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(cfg.epochs):
        loss = train_epoch(model, train_loader, optimizer, cfg)
        print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
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

    # 对每只股票单独回测并汇总指标
    summary = []

    for code, df, test_ds, scaler, feature_cols, extrema_info, train_indices, test_indices in test_info:
        if len(test_indices) == 0:
            print(f"股票 {code} 测试信号为空，跳过")
            continue

        test_start_loc = max(min(test_indices), cfg.window_size)
        test_end_loc = len(df)

        # 1. 用训练期预测分布校准阈值
        threshold = calibrate_threshold_from_train(
            df, model_eval, cfg, scaler, feature_cols, train_indices
        )

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

        # 5. 打印模型预测分布
        if buy_records:
            df_rec = pd.DataFrame(buy_records)
            print(f"\n===== 股票 {code} 测试集买入信号 d_mid 分析 =====")
            print(f"测试集买入信号数: {len(df_rec)}")
            print(f"threshold: {threshold:.4f}")
            print(f"d_mid 最小值: {df_rec['d_mid'].min():.4f}")
            print(f"d_mid 最大值: {df_rec['d_mid'].max():.4f}")
            print(f"d_mid 中位数: {df_rec['d_mid'].median():.4f}")

            executed_count = int(df_rec['executed'].sum())
            print(f"实际执行买入信号数: {executed_count}")

            if executed_count > 0:
                print("执行的买入信号详情:")
                print(df_rec[df_rec['executed']][['time', 'price', 'd_mid', 'threshold']].to_string(index=False))
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

        sharpe_b = calc_sharpe(equity_b)
        mdd_b = calc_max_drawdown(equity_b)

        sharpe_o = calc_sharpe(equity_o)
        mdd_o = calc_max_drawdown(equity_o)

        summary.append((code, sharpe_f, mdd_f, sharpe_b, mdd_b, sharpe_o, mdd_o))

        print(
            f"\n股票 {code}: "
            f"CTAML过滤 夏普={sharpe_f:.4f}, MDD={mdd_f:.4%} | "
            f"无过滤 夏普={sharpe_b:.4f}, MDD={mdd_b:.4%} | "
            f"Oracle标签 夏普={sharpe_o:.4f}, MDD={mdd_o:.4%}"
        )

        analyze_prediction_vs_label(df, buy_records, good_threshold=cfg.oracle_D_threshold)

    if summary:
        avg_sharpe_f = np.mean([x[1] for x in summary])
        avg_mdd_f = np.mean([x[2] for x in summary])

        avg_sharpe_b = np.mean([x[3] for x in summary])
        avg_mdd_b = np.mean([x[4] for x in summary])

        avg_sharpe_o = np.mean([x[5] for x in summary])
        avg_mdd_o = np.mean([x[6] for x in summary])

        print("\n===== 汇总结果 =====")
        print(f"CTAML过滤: 平均夏普={avg_sharpe_f:.4f}, 平均MDD={avg_mdd_f:.4%}")
        print(f"无过滤:     平均夏普={avg_sharpe_b:.4f}, 平均MDD={avg_mdd_b:.4%}")
        print(f"Oracle标签: 平均夏普={avg_sharpe_o:.4f}, 平均MDD={avg_mdd_o:.4%}")


if __name__ == "__main__":
    main()
