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
    evt_quantile = 0.95  # 极端事件的收益率绝对值分位数阈值
    extrema_win = 1  # 极值点邻域半径（日）
    # 对比学习
    contrastive_win = 5  # 对比学习正样本窗口半径  正/负样本的环境窗口半径（论文取±5步）
    D_buckets = 8  # 样本均衡分桶数（此处预留，用于按D值分层采样）
    D_threshold = 0.4  # 信号过滤阈值：D中位数小于此值才执行信号

    # 模型
    encoder_type = 'Informer'  # 'BiLSTM' 或 'Informer'
    d_model = 128  # 隐藏层/嵌入维度
    n_heads = 8  # 多头注意力的头数
    proj_dim = 128  # 对比学习投影头输出维度
    quantiles = [0.1, 0.5, 0.9]  # 分位数回归的分位点
    use_taa = True  # 是否启用尾部感知注意力模块
    lam_init = 0.1  # TAA中可学习缩放参数λ的初始值
    tau_cl = 0.1  # InfoNCE损失的温度系数
    # 多任务损失权重
    lambda_D = 1.0
    lambda_xi = 0.5
    lambda_cl = 0.3

    # 训练
    batch_size = 64
    lr = 1e-3
    epochs = 100
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
def compute_evt_distance_xi(df, cfg):
    """
    核心标注函数：
    1. 基于滚动窗口拟合GPD，产生时变的形状参数ξ标签；
    2. 识别结构性极值点；
    3. 为每个信号点计算归一化时空距离D。
    返回：扩充了D_label和xi_label的DataFrame，以及全时间轴的ξ、σ序列。
    """
    # 提取收益率及其绝对值，转为numpy数组便于快速索引
    returns = df['returns'].values
    abs_r = df['abs_returns'].values
    n = len(df)

    # 1. 基于滚动窗口拟合GPD，产生时变的形状参数ξ标签；
    # 初始化ξ和σ序列（全NaN），只对被标记为极端事件的时间步赋值
    xi_series = np.full(n, np.nan)
    sigma_series = np.full(n, np.nan)
    # 滚动计算收益率绝对值的95%分位数作为动态阈值
    threshold_series = pd.Series(abs_r).rolling(cfg.evt_rolling, min_periods=cfg.evt_rolling).quantile(
        cfg.evt_quantile).values

    # 对每个时间点，用过去252日的超阈值收益拟合GPD，得到时变的ξ和σ
    for i in range(cfg.evt_rolling, n):  # 从第 cfg.evt_rolling（配置文件中的滚动窗口，默认252天，约一年）天开始，遍历到数据结束。
        # 提取当前时间点i之前 252 天的收益率绝对值。因为我们只关心“波动的剧烈程度”（无论是暴涨还是暴跌），所以取绝对值。这是为了捕捉市场的整体尾部风险水平。
        window_abs = abs_r[i - cfg.evt_rolling:i]

        # 获取当前时间点i对应的动态阈值.这个阈值通常是过去252天收益率绝对值的 95% 分位数（u）。只有超过这个阈值的波动，才被定义为“极端事件”。
        thresh = threshold_series[i]

        """
        核心：筛选超阈值数据 (Peaks Over Threshold, POT)
        含义：这是 EVT（极值理论）中最关键的一步。
            1、window_abs[window_abs > thresh]：筛选出那些超过阈值u的极端波动数据。
            2、- thresh：将这些极端数据减去阈值u，得到“超出量”（Excesses）。
        数学逻辑：根据 Pickands–Balkema–de Haan 定理，当阈值u足够大时，超出阈值的数据的分布收敛于广义帕累托分布 (GPD)。
        目的：我们要拟合的就是这些“超出量”的分布，而不是原始收益率的分布。
        """
        exceed = window_abs[window_abs > thresh] - thresh  # 超出部分

        """
        含义：如果超过阈值的极端事件少于 10 个，则跳过本次拟合。
        逻辑：统计学常识。如果极端事件太少（比如只有1-2个），拟合出来的参数会非常不稳定（方差极大），没有统计意义。10个是一个经验阈值。
        """
        if len(exceed) >= 10:  # 保证有足够样本点进行稳定拟合
            try:
                """
                含义：调用 scipy 库中的 genpareto.fit 函数，使用极大似然估计 (MLE) 方法，计算 exceed 这组数据最符合的 GPD 分布参数。
                    shape (形状参数)：这就是论文中反复提到的ξ (Xi)。它决定了尾部的厚度。
                    scale (尺度参数)：这就是σ (Sigma)。它决定了超出量的平均大小。
                    _：位置参数，在 GPD 拟合中通常固定为 0 或忽略，所以用下划线接收。
                """
                shape, _, scale = genpareto.fit(exceed)  # shape即为ξ，scale为σ

                """
                含义：将当前时间点i 拟合出来的ξ和σ存入序列中。
                结果：最终你会得到两条随时间变化的曲线：
                    xi_series：记录了市场在每个时刻的“尾部风险程度”。
                    sigma_series：记录了市场在每个时刻的“波动幅度”。
                """
                xi_series[i] = shape
                sigma_series[i] = scale
            except:
                pass  # 拟合失败则保留NaN，后续会被过滤

    # 识别异常，识别极端事件掩码：收益率绝对值超过动态阈值
    """
    代码含义：生成一个“真假列表”。
    逻辑：abs_r（收益率绝对值）如果超过了 threshold_series（动态阈值，通常是 95% 分位数），就标记为 True（异常），否则为 False。
    形象比喻：这就像是在 1000 天的行情里，用一个筛子把那些“疯涨”或“疯跌”的日子先挑了出来。
    """
    extreme_mask = abs_r > threshold_series

    # 2. 识别结构性极值点；
    # 结构性极值点定位：在极端事件日前后各1天窗口内找局部最高/最低点
    extrema_idx_set = set()
    for i in np.where(extreme_mask)[0]:
        """
        代码含义：
            1、遍历每一个“极端事件日”（idx）。
            2、关键动作：它不直接把“极端事件日”当天当作顶/底，而是以这一天为中心，向前后各延伸 1 天（即 idx-1 到 idx+1），形成一个 3 天的小窗口。
            3、在这 3 天的 close 价格中找最高点或最低点。
            为什么要这么做？（论文级解释）
                1、现实市场的滞后性：在金融市场上，价格剧烈波动（极端收益率）往往发生在趋势的末端。比如，今天暴跌了 6%（极端事件），但真正的
                “谷底”可能就是今天收盘，或者因为恐慌延续到明天早盘。
                2、避免未来函数（Look-ahead Bias）：如果你直接用“极端事件日”当天作为极值点，模型可能会学到错误的信号。
                通过在 [-1, +1] 的窗口内寻找局部最高/最低点，你定义了一个包含价格惯性的“极值区域”，这比单点更稳健，也更符合交易直觉（即：这个极端波动代表了一个局部的顶或底）。
        
        不必担心窗口小：连续暴跌中的每一天都是独立的极端事件日，系统会为每一天生成局部极值候选，最终通过后处理保留全局最低点。
        """
        if returns[i] > 0:
            # 寻找邻域最高点
            local_start = max(0, i - cfg.extrema_win)
            local_end = min(n - 1, i + cfg.extrema_win)
            local_close = df['close'].iloc[local_start:local_end + 1]
            t_max = local_close.idxmax()  # 时间戳
            idx_ext = df.index.get_loc(t_max)  # 整数位置
            extrema_idx_set.add(idx_ext)
        else:
            # 邻域最低点
            local_start = max(0, i - cfg.extrema_win)
            local_end = min(n - 1, i + cfg.extrema_win)
            local_close = df['close'].iloc[local_start:local_end + 1]
            t_min = local_close.idxmin()
            idx_ext = df.index.get_loc(t_min)
            extrema_idx_set.add(idx_ext)

    # 将极值点的时间索引转换为位置索引，存入集合以便快速查找
    """
    代码含义：把刚才找到的所有极值点的时间戳，转换成数据框里的“行号”（位置索引），并存成一个集合（Set）。
    逻辑：
        df.index.get_loc(t)：把时间（比如 '2026-05-09'）转换成行号（比如第 500 行）。
        set([...])：使用集合（Set）而不是列表（List），是为了让后续的“查找”操作变得极快（时间复杂度 O(1)）。
    用途：在后续的代码中（比如计算距离D或对比学习负采样时），模型需要频繁地问：“这个时间点是不是极值点？” 使用这个 extrema_idx 集合，电脑可以瞬间回答“是”或“不是”，而不需要从头到尾去遍历整个数据表。
    """
    # 3. 遍历所有信号点，计算其对应的距离标签D
    D_labels = []  # 距离标签 → 信号质量分数（越小越好，0=完美信号，1=无效信号）
    xi_labels = []  # 辅助标签 → 极端事件尾部厚度（用于双任务学习）

    # 作用：只处理有交易信号的日子（如均线金叉/死叉），跳过空仓期。
    signal_times = df[df['signal'] != 'none'].index
    for t_s in signal_times:
        idx_s = df.index.get_loc(t_s)
        # 搜索未来窗口内的极值点（按 bar 数）
        future_end = min(n, idx_s + cfg.future_T + 1)
        min_dist = 1.0  # 默认最差质量
        # 寻找第一个出现的结构性极值点
        """
        连续暴跌中，第一个极值点是否可靠？
            是的，且这是设计精髓。原因有三：
            1、金融市场的“首次转折”最具交易价值
            交易者不会等待“最低点”，而是在趋势首次确认反转时行动。例如：
            连续暴跌4天：第2天创新低 → 第3天反弹 → 第4天再创新低
            第一个极值点（第2天）是趋势首次衰竭信号，第4天的“更低点”已是二次探底（风险收益比更差）。
            实盘意义：模型学到的是“趋势首次转折的捕捉能力”，而非“猜最低点”（后者是过度拟合）。
            2、避免未来函数污染
            如果取“最近极值点”，在连续暴跌中可能选到第4天的谷底（但第2天时第4天数据未知）。
            只取“第一个”保证了实时性：信号发出时，模型只能利用已发生的转折点。
            3、符合 EVT 理论的“超越点”定义
            广义帕累托分布（GPD）建模的是首次突破阈值后的超额分布。
            在你的框架中，extreme_mask 标记了“阈值突破日”，而第一个极值点就是首次突破后的结构性确认点。
        """
        for idx_t in range(idx_s + 1, future_end):
            if idx_t in extrema_idx_set:  # extrema_idx_set 是极值点的位置集合
                P_ext = df['close'].iloc[idx_t]
                P_s = df['close'].iloc[idx_s]
                atr_s = df['atr'].iloc[idx_s]  # 信号点处的ATR
                time_diff = idx_t - idx_s  # 时间差用 bar 数
                price_diff = abs(P_ext - P_s)
                # 归一化时空距离，对应论文公式(1)
                """
                时间维度	(time_diff / future_T)²	信号到极值点的时间成本	将时间压缩到 [0,1]：第1天到达=0.1，第10天到达=1.0（假设 future_T=10）
                价格维度	(price_diff / (kappa * atr_s))²	信号到极值点的价格空间（用ATR标准化波动率）	ATR 是动态波动率指标，kappa（通常=2~3）确保价格项与时间项量级相当
                
                为什么用 ATR 而非固定价格阈值？
                    固定阈值问题：10元股跌1元=10%，100元股跌1元=1% → 无法跨品种比较。
                    ATR 解决方案：
                    price_diff / (kappa * atr_s) = 价格变动占当前波动率的比例
                    例：ATR=2元时，价格变动4元 = 2倍ATR（高波动环境中的显著变动）
                    例：ATR=0.5元时，价格变动4元 = 8倍ATR（极端事件，需重点关注）
                
                为什么用 欧氏距离而非简单加权？
                    几何意义：将“时间-价格”视为二维平面，距离越短代表信号越高效。
                """
                dist = np.sqrt((time_diff / cfg.future_T) ** 2 + (price_diff / (cfg.kappa * atr_s)) ** 2)
                min_dist = min(min_dist, dist)
                break  # 只取第一个极值点
        D_labels.append(min_dist)
        xi_labels.append(xi_series[idx_s])  # 该信号点的当前 ξ

    # 合并到df
    df_out = df.copy()
    df_out['D_label'] = np.nan
    df_out['xi_label'] = np.nan
    locs = [df.index.get_loc(t) for t in signal_times]
    df_out.iloc[locs, df_out.columns.get_loc('D_label')] = D_labels
    df_out.iloc[locs, df_out.columns.get_loc('xi_label')] = xi_labels
    return df_out, xi_series, sigma_series, extrema_idx_set


# ================== 3. 数据集与对比学习准备 ==================
class SignalDataset(Dataset):
    """
    三元组数据集：为每个交易信号点提供 (信号窗口, 未来极值窗口[正样本], 远离极值窗口[负样本])。
    支持在线标准化、样本均衡和对比学习的正负样本生成。
    """

    def __init__(self, df, feature_cols, cfg, extrema_idx_set, scaler=None, is_train=True):
        self.df = df
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels_D = df['D_label'].values
        self.labels_xi = df['xi_label'].values
        self.signal_mask = (df['signal'] != 'none').values  # 布尔掩码，标识信号点
        self.window = cfg.window_size
        self.cfg = cfg
        # 由于compute_evt_distance_xi函数改为用bar位置计算距离而非时间，因此此处需要确保 extrema_idx_set 传入
        self.extrema_idx_set = extrema_idx_set  # 返回位置集合

        # 特征标准化：训练集拟合scaler，测试/验证集直接使用传入的scaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

        # 筛选有效信号点（保证正样本窗口不越界）
        raw_indices = np.where(self.signal_mask)[0]
        self.signal_indices = [
            i for i in raw_indices
            if i >= self.window
               and i + cfg.future_T + cfg.contrastive_win < len(df)  # 关键：给正样本窗口留足空间
               and not np.isnan(self.labels_D[i])
               and not np.isnan(self.labels_xi[i])
        ]

    def __len__(self):
        """返回有效信号点的数量"""
        return len(self.signal_indices)

    def __getitem__(self, idx):
        """返回单个样本的6项数据：信号窗口、D标签、ξ标签、收益率绝对值序列、正样本窗口、负样本窗口"""
        i = self.signal_indices[idx]  # 信号点在特征数组中的绝对位置索引
        # 提取前window步的特征窗口 (W, F)
        x_win = self.features[i - self.window:i]  # (W, F)
        d_label = self.labels_D[i]
        xi_label = self.labels_xi[i]
        # 提取该窗口内的收益率绝对值序列，供TAA使用
        abs_ret = self.df['abs_returns'].values[i - self.window:i]

        # 正样本窗口：未来第一个极值点前后各contrastive_win天的环境窗口

        # 若未来窗口内未找到极值点，用全零张量占位（实际训练中会被掩盖或忽略）
        pos_win = np.zeros((2 * self.cfg.contrastive_win + 1, self.features.shape[1]), dtype=np.float32)
        future_end = min(len(self.df), i + cfg.future_T + 1)
        for j in range(i + 1, future_end):
            if j in self.extrema_idx_set:  # 找到第一个结构性极值点
                start = max(0, j - cfg.contrastive_win)
                end = min(len(self.df), j + cfg.contrastive_win + 1)
                # 得益于筛选条件，这里 start:end 长度必定 >= 2*contrastive_win+1
                pos_win = self.features[start:end]  # 如果仍想安全，可做一次裁剪/填充
                break

        # 负样本窗口（随机远离极值点的区域） （保证长度固定）
        neg_win = np.zeros_like(pos_win)
        candidates = []
        for _ in range(10):
            k = np.random.randint(self.window, len(self.df) - 2 * self.cfg.contrastive_win)
            if all(abs(k - e) > self.cfg.future_T / 2 for e in self.extrema_idx_set):
                candidates.append(k)
        if candidates:
            k = np.random.choice(candidates)
            neg_win = self.features[k: k + 2 * self.cfg.contrastive_win + 1]

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
        z = abs_returns / (sigma_hat + 1e-8)  # abs_returns: (B, T)
        log_surv = -1.0 / (xi_hat + 1e-6) * torch.log1p(xi_hat * z + 1e-8)  # (B, T)
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
        loss_cl = info_nce(proj_anchor, proj_pos, cfg.tau_cl)

        loss = cfg.lambda_D * loss_D + cfg.lambda_xi * loss_xi + cfg.lambda_cl * loss_cl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ================== 7. 回测与决策 ==================
def apply_filter_and_trade(df, model, cfg, scaler, feature_cols):
    signal_idx = df[df['signal'] != 'none'].index
    capital = 1.0
    position = 0
    equity = [capital]
    for idx, row in df.iterrows():
        if idx in signal_idx:
            i_loc = df.index.get_loc(idx)
            if i_loc >= cfg.window:
                x_win = scaler.transform(df[feature_cols].iloc[i_loc-cfg.window:i_loc].values)
                x_ten = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                abs_ret = torch.tensor(df['abs_returns'].iloc[i_loc-cfg.window:i_loc].values,
                                       dtype=torch.float32).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    D_hat, _, _ = model(x_ten, abs_ret,
                                        torch.zeros(1).to(cfg.device),
                                        torch.tensor([1.0]).to(cfg.device))
                    d_mid = D_hat[0,1].item()
                if d_mid < cfg.D_threshold and row['signal'] == 'buy':
                    position = capital / row['close']
                    capital = 0
                elif row['signal'] == 'sell' and position > 0:
                    capital = position * row['close']
                    position = 0
        if position > 0:
            equity.append(position * row['close'])
        else:
            equity.append(capital)
    return equity


def labeled_signals(code_short, df, signal_path):
    # 将 D_label 和 xi_label 写回 signals.csv 对应的行
    signals_orig = pd.read_csv(signal_path, parse_dates=['time'])
    # 从标注结果中取出信号点的标签
    labeled_signals = df[df['signal'] != 'none'][['D_label', 'xi_label']].copy()
    labeled_signals.index.name = 'time'
    labeled_signals = labeled_signals.reset_index()
    # 按时间合并
    signals_labeled = signals_orig.merge(labeled_signals, on='time', how='left')
    signals_labeled.to_csv("./trade_point_backtest_tea_radical_nature/"+code_short+"signals_labeled.csv", index=False)


# ================== 8. 主程序（示例） ==================
def main():
    print("加载数据...")
    stocks_df = pd.read_csv("D://github//RobotMeQ//QuantData//asset_code//a800_stocks_2025.csv", dtype={'code': str})
    for _, row in stocks_df.iterrows():
        code = row['code']  # 如 'sh.600000'
        # 构造文件路径，注意你的命名规则
        code_short = code.split('.')[-1]  # '600000'
        ohlcv_path = f"D:/github/RobotMeQ_Dataset/QuantData/backTest/bar_A_{code_short}_d.csv"
        signal_path = f"D:/github/RobotMeQ_Dataset/QuantData/trade_point_backtest_tea_radical_nature/A_{code_short}_d.csv"

        df, feature_cols = load_and_prepare(signal_path, ohlcv_path)
        print(code+"计算极值标注...")
        df, xi_series, sigma_series, extrema_idx_set = compute_evt_distance_xi(df, cfg)
        # df.to_csv(code+"_labeled_data.csv")  # 保存标注结果
        labeled_signals(code_short, df, signal_path)
    # print("构建数据集...")
    # dataset = SignalDataset(df, feature_cols, cfg, extrema_idx_set, scaler=None)
    # scaler = dataset.scaler
    # dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    #
    # print("初始化编码器...")
    # if cfg.encoder_type == 'BiLSTM':
    #     encoder = BiLSTMEncoder(input_dim=len(feature_cols), d_model=cfg.d_model)
    # elif cfg.encoder_type == 'Informer':
    #     encoder = InformerEncoder(input_dim=len(feature_cols), d_model=cfg.d_model,
    #                               seq_len=cfg.window_size, n_heads=cfg.n_heads)
    # else:
    #     raise ValueError("Unknown encoder_type")
    #
    # model = CTAML(encoder, d_model=cfg.d_model, n_heads=cfg.n_heads,
    #               proj_dim=cfg.proj_dim, quantiles=cfg.quantiles,
    #               use_taa=cfg.use_taa).to(cfg.device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    #
    # print("开始训练...")
    # best_loss = float('inf')
    # patience_counter = 0
    # for epoch in range(cfg.epochs):
    #     loss = train_epoch(model, dataloader, optimizer, cfg)
    #     print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {loss:.4f}")
    #     if loss < best_loss:
    #         best_loss = loss
    #         patience_counter = 0
    #         torch.save(model.state_dict(), "best_ctaml.pth")
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= cfg.patience:
    #             print("早停触发")
    #             break
    #
    # print("训练完成，模型已保存。可运行回测。")


if __name__ == "__main__":
    main()
