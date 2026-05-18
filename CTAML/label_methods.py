"""
label_methods.py

通用 Meta-Labeling 标注方法模块。

设计目标
--------
1. 将 CTAML_eswa_main.py 中和具体标注方法相关的代码抽离出来；
2. 为不同标注方法提供统一 API；
3. 所有标注方法最终都输出两个标准列：
   - label_value: 连续或离散标签值，可用于诊断；
   - meta_label:  二分类元标签，1 表示高质量 buy 信号，0 表示低质量 buy 信号。

主框架 CTAML_eswa_main.py 只依赖这两个标准列，
因此你可以轻松替换 ERDF、Triple Barrier、未来收益、固定极值等标注方法。
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto


# ============================================================
# 1. 通用工具函数
# ============================================================

def target_extrema_type(signal_type):
    """
    对于 long-only 策略：
    - buy 信号理想上应靠近未来结构性低点 valley；
    - sell 信号理想上应靠近未来结构性高点 peak。

    当前主框架只训练 buy 过滤器，但这里仍然保留 sell 逻辑，
    方便以后扩展 sell-side meta-labeling。
    """
    if signal_type == 'buy':
        return 'valley'
    elif signal_type == 'sell':
        return 'peak'
    else:
        return None


def ensure_standard_label_columns(df, label_value_col="label_value", meta_label_col="meta_label"):
    """
    确保 DataFrame 中存在标准标签列。
    """
    df = df.copy()

    if label_value_col not in df.columns:
        df[label_value_col] = np.nan

    if meta_label_col not in df.columns:
        df[meta_label_col] = np.nan

    return df


def print_label_diagnostics(
        code,
        df,
        signal_indices=None,
        label_value_col="label_value",
        meta_label_col="meta_label",
        prefix=""
):
    """
    通用标签诊断函数。

    参数
    ----
    code : str
        股票代码。
    df : pd.DataFrame
        已经打好标签的数据。
    signal_indices : list[int] or None
        需要诊断的信号位置。如果为 None，则诊断所有非 none 信号。
    label_value_col : str
        连续标签列名称。
    meta_label_col : str
        二元标签列名称。
    prefix : str
        打印前缀，例如“训练期”、“验证期”、“测试期”。
    """
    if signal_indices is None:
        sig_df = df[df['signal'] != 'none'].copy()
    else:
        sig_df = df.iloc[signal_indices]
        sig_df = sig_df[sig_df['signal'] != 'none'].copy()

    if len(sig_df) == 0:
        print(f"{prefix}股票 {code}: 无信号可诊断")
        return

    if label_value_col not in sig_df.columns or meta_label_col not in sig_df.columns:
        print(f"{prefix}股票 {code}: 缺少标签列 {label_value_col} 或 {meta_label_col}")
        return

    valid = sig_df.dropna(subset=[label_value_col, meta_label_col])

    if len(valid) == 0:
        print(f"{prefix}股票 {code}: 信号标签全为空")
        return

    v = valid[label_value_col].astype(float)
    y = valid[meta_label_col].astype(float)

    print(f"\n===== {prefix}股票 {code} 标签诊断 =====")
    print(f"信号数: {len(valid)}")
    print(
        f"{label_value_col} min/median/mean/max: "
        f"{v.min():.4f} / {v.median():.4f} / {v.mean():.4f} / {v.max():.4f}"
    )
    print(f"meta_label=1 数量: {int(y.sum())}, 占比: {y.mean():.2%}")
    print(
        valid.groupby('signal')[[label_value_col, meta_label_col]]
        .agg(['count', 'mean', 'median', 'min', 'max'])
        .to_string()
    )


# ============================================================
# 2. 标注器基类
# ============================================================

class BaseLabeler:
    """
    所有标注方法的基类。

    每个子类必须实现 transform(df, cfg)，并返回：
        df_out, info

    其中 df_out 至少包含：
        - cfg.label_value_col
        - cfg.meta_label_col

    info 是字典，用于返回标注过程中的额外信息，例如极值点、xi 序列等。
    """

    name = "base"

    def transform(self, df, cfg):
        raise NotImplementedError("子类必须实现 transform(df, cfg)")


# ============================================================
# 3. ERDF-EVT 标注方法
# ============================================================

class ERDFEVTLabeler(BaseLabeler):
    """
    ERDF-EVT 标签。

    核心逻辑
    --------
    1. 使用滚动高分位阈值识别极端收益事件；
    2. 在极端收益事件附近寻找结构性 peak / valley；
    3. 对每个信号计算到未来目标极值区域的归一化时空距离 D；
    4. 对 buy 信号，如果 D <= cfg.good_label_threshold，则 meta_label=1。

    输出
    ----
    cfg.label_value_col : D 距离标签，越小越好；
    cfg.meta_label_col  : 二分类标签；
    xi_label            : GPD shape 参数，当前主框架不使用，但保留给扩展模型。
    """

    name = "erdf_evt"

    def transform(self, df, cfg):
        df_out = ensure_standard_label_columns(
            df,
            label_value_col=cfg.label_value_col,
            meta_label_col=cfg.meta_label_col
        )

        returns = df_out['returns'].values
        abs_r = df_out['abs_returns'].values
        n = len(df_out)

        xi_series = np.full(n, np.nan)
        sigma_series = np.full(n, np.nan)

        # 1. 滚动 EVT 阈值
        threshold_series = pd.Series(abs_r).rolling(
            cfg.evt_rolling,
            min_periods=cfg.evt_rolling
        ).quantile(cfg.evt_quantile).values

        # 2. 滚动 GPD 拟合，得到 xi 和 sigma
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

        xi_series = pd.Series(xi_series).ffill().bfill().fillna(0.0).values
        sigma_series = (
            pd.Series(sigma_series)
            .ffill()
            .bfill()
            .fillna(np.nanstd(abs_r) + 1e-6)
            .values
        )

        # 3. 根据极端收益事件定位结构性极值
        extreme_mask = (abs_r > threshold_series) & (~np.isnan(threshold_series))
        extrema_info = {}

        for i in np.where(extreme_mask)[0]:
            local_start = max(0, i - cfg.extrema_win)
            local_end = min(n - 1, i + cfg.extrema_win)
            local_close = df_out['close'].iloc[local_start:local_end + 1]

            if len(local_close) == 0:
                continue

            if returns[i] > 0:
                # 正极端收益附近找结构性高点
                t_max = local_close.idxmax()
                idx_ext = df_out.index.get_loc(t_max)
                extrema_info[idx_ext] = 'peak'
            else:
                # 负极端收益附近找结构性低点
                t_min = local_close.idxmin()
                idx_ext = df_out.index.get_loc(t_min)
                extrema_info[idx_ext] = 'valley'

        # 4. 为所有非 none 信号计算 D 距离
        signal_times = df_out[df_out['signal'] != 'none'].index

        label_values = []
        meta_labels = []
        xi_labels = []

        for t_s in signal_times:
            idx_s = df_out.index.get_loc(t_s)
            signal_type = df_out['signal'].loc[t_s]
            target_type = target_extrema_type(signal_type)

            future_end = min(n, idx_s + cfg.future_T + 1)
            min_dist = 1.0

            if target_type is not None:
                for idx_t in range(idx_s, future_end):
                    if idx_t not in extrema_info:
                        continue

                    if extrema_info[idx_t] != target_type:
                        continue

                    P_ext = df_out['close'].iloc[idx_t]
                    P_s = df_out['close'].iloc[idx_s]
                    atr_s = df_out['atr'].iloc[idx_s]

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

            label_value = min_dist

            # 当前主框架只训练 buy 信号。
            # 对 buy 信号而言，D 越小越好。
            if signal_type == cfg.train_signal_type:
                meta_label = 1.0 if label_value <= cfg.good_label_threshold else 0.0
            else:
                meta_label = np.nan

            label_values.append(label_value)
            meta_labels.append(meta_label)
            xi_labels.append(xi_series[idx_s])

        locs = [df_out.index.get_loc(t) for t in signal_times]

        df_out.iloc[locs, df_out.columns.get_loc(cfg.label_value_col)] = label_values
        df_out.iloc[locs, df_out.columns.get_loc(cfg.meta_label_col)] = meta_labels

        df_out['xi_label'] = np.nan
        df_out.iloc[locs, df_out.columns.get_loc('xi_label')] = xi_labels

        info = {
            "labeler": self.name,
            "xi_series": xi_series,
            "sigma_series": sigma_series,
            "extrema_info": extrema_info,
            "label_value_col": cfg.label_value_col,
            "meta_label_col": cfg.meta_label_col
        }

        return df_out, info


# ============================================================
# 4. 固定局部极值标注方法
# ============================================================

class FixedExtremaLabeler(BaseLabeler):
    """
    固定局部极值标签。

    不使用 EVT，只在固定窗口内寻找普通局部 high / low。
    该方法适合作为一种实验标签，而不是默认外部 Oracle 基线。
    """

    name = "fixed_extrema"

    def transform(self, df, cfg):
        df_out = ensure_standard_label_columns(
            df,
            label_value_col=cfg.label_value_col,
            meta_label_col=cfg.meta_label_col
        )

        n = len(df_out)
        close = df_out['close'].values
        w = cfg.fixed_extrema_win

        extrema_info = {}

        for i in range(w, n - w):
            local = close[i - w:i + w + 1]

            if close[i] == np.min(local):
                extrema_info[i] = 'valley'
            elif close[i] == np.max(local):
                extrema_info[i] = 'peak'

        signal_times = df_out[df_out['signal'] != 'none'].index

        label_values = []
        meta_labels = []

        for t_s in signal_times:
            idx_s = df_out.index.get_loc(t_s)
            signal_type = df_out['signal'].loc[t_s]
            target_type = target_extrema_type(signal_type)

            future_end = min(n, idx_s + cfg.future_T + 1)
            min_dist = 1.0

            if target_type is not None:
                for idx_t in range(idx_s, future_end):
                    if idx_t not in extrema_info:
                        continue
                    if extrema_info[idx_t] != target_type:
                        continue

                    P_ext = df_out['close'].iloc[idx_t]
                    P_s = df_out['close'].iloc[idx_s]
                    atr_s = df_out['atr'].iloc[idx_s]

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

            label_value = min_dist

            if signal_type == cfg.train_signal_type:
                meta_label = 1.0 if label_value <= cfg.good_label_threshold else 0.0
            else:
                meta_label = np.nan

            label_values.append(label_value)
            meta_labels.append(meta_label)

        locs = [df_out.index.get_loc(t) for t in signal_times]
        df_out.iloc[locs, df_out.columns.get_loc(cfg.label_value_col)] = label_values
        df_out.iloc[locs, df_out.columns.get_loc(cfg.meta_label_col)] = meta_labels

        info = {
            "labeler": self.name,
            "extrema_info": extrema_info,
            "label_value_col": cfg.label_value_col,
            "meta_label_col": cfg.meta_label_col
        }

        return df_out, info


# ============================================================
# 5. 未来收益标注方法示例
# ============================================================

class FutureReturnLabeler(BaseLabeler):
    """
    未来收益标签。

    适合作为最简单的对照标注方法。

    label_value = future_T 日后收益率
    meta_label = 1 if future_return >= cfg.future_return_threshold else 0

    注意：
    - 这种标签直接追求收益；
    - 与 ERDF 标签的“靠近结构性低点”逻辑不同。
    """

    name = "future_return"

    def transform(self, df, cfg):
        df_out = ensure_standard_label_columns(
            df,
            label_value_col=cfg.label_value_col,
            meta_label_col=cfg.meta_label_col
        )

        n = len(df_out)
        signal_times = df_out[df_out['signal'] != 'none'].index

        label_values = []
        meta_labels = []

        for t_s in signal_times:
            idx_s = df_out.index.get_loc(t_s)
            signal_type = df_out['signal'].loc[t_s]

            if idx_s + cfg.future_T >= n:
                label_value = np.nan
                meta_label = np.nan
            else:
                p0 = df_out['close'].iloc[idx_s]
                p1 = df_out['close'].iloc[idx_s + cfg.future_T]

                if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
                    label_value = np.nan
                    meta_label = np.nan
                else:
                    label_value = p1 / p0 - 1.0

                    if signal_type == cfg.train_signal_type:
                        meta_label = 1.0 if label_value >= cfg.future_return_threshold else 0.0
                    else:
                        meta_label = np.nan

            label_values.append(label_value)
            meta_labels.append(meta_label)

        locs = [df_out.index.get_loc(t) for t in signal_times]
        df_out.iloc[locs, df_out.columns.get_loc(cfg.label_value_col)] = label_values
        df_out.iloc[locs, df_out.columns.get_loc(cfg.meta_label_col)] = meta_labels

        info = {
            "labeler": self.name,
            "label_value_col": cfg.label_value_col,
            "meta_label_col": cfg.meta_label_col
        }

        return df_out, info


# ============================================================
# 6. Triple Barrier 作为模型训练标签的可选实现
# ============================================================

class TripleBarrierMetaLabeler(BaseLabeler):
    """
    Triple Barrier 标注方法。

    这个类用于“让序列模型也使用 Triple Barrier 标签训练”。
    注意它和主框架里的 Triple Barrier + XGB 外部基线不同：
    - 这里输出 meta_label 给 BiLSTM 模型训练；
    - 外部基线则是 Triple Barrier 标签 + XGBoost。
    """

    name = "triple_barrier_meta"

    def transform(self, df, cfg):
        df_out = ensure_standard_label_columns(
            df,
            label_value_col=cfg.label_value_col,
            meta_label_col=cfg.meta_label_col
        )

        n = len(df_out)
        signal_times = df_out[df_out['signal'] != 'none'].index

        label_values = []
        meta_labels = []

        for t_s in signal_times:
            idx_s = df_out.index.get_loc(t_s)
            signal_type = df_out['signal'].loc[t_s]

            if signal_type != cfg.train_signal_type:
                label_values.append(np.nan)
                meta_labels.append(np.nan)
                continue

            price0 = df_out['close'].iloc[idx_s]
            atr0 = df_out['atr'].iloc[idx_s]

            if not np.isfinite(price0) or not np.isfinite(atr0) or atr0 <= 1e-12:
                label_values.append(np.nan)
                meta_labels.append(np.nan)
                continue

            upper = price0 + cfg.tb_pt_atr * atr0
            lower = price0 - cfg.tb_sl_atr * atr0
            end = min(n, idx_s + cfg.tb_horizon + 1)

            y = 0.0
            first_hit = 0.0

            for j in range(idx_s + 1, end):
                high_j = df_out['high'].iloc[j]
                low_j = df_out['low'].iloc[j]

                hit_upper = high_j >= upper
                hit_lower = low_j <= lower

                if hit_upper and hit_lower:
                    y = 0.0
                    first_hit = -1.0
                    break
                elif hit_upper:
                    y = 1.0
                    first_hit = 1.0
                    break
                elif hit_lower:
                    y = 0.0
                    first_hit = -1.0
                    break

            label_values.append(first_hit)
            meta_labels.append(y)

        locs = [df_out.index.get_loc(t) for t in signal_times]
        df_out.iloc[locs, df_out.columns.get_loc(cfg.label_value_col)] = label_values
        df_out.iloc[locs, df_out.columns.get_loc(cfg.meta_label_col)] = meta_labels

        info = {
            "labeler": self.name,
            "label_value_col": cfg.label_value_col,
            "meta_label_col": cfg.meta_label_col
        }

        return df_out, info


# 自定义，60bar内涨超20%
class MyUp20PercentLabeler(BaseLabeler):
    name = "up20pct"

    def transform(self, df, cfg):
        df_out = ensure_standard_label_columns(df, cfg.label_value_col, cfg.meta_label_col)
        n = len(df_out)

        signal_times = df_out[df_out['signal'] != 'none'].index
        label_values = []
        meta_labels = []

        for t_s in signal_times:
            idx_s = df_out.index.get_loc(t_s)
            if idx_s + cfg.future_T >= n:
                label_values.append(np.nan)
                meta_labels.append(np.nan)
                continue

            # 未来60根K线的最高价
            future_high = df_out['high'].iloc[idx_s: idx_s + cfg.future_T + 1].max()
            p0 = df_out['close'].iloc[idx_s]
            max_rise = future_high / p0 - 1.0   # 最大涨幅

            label_value = max_rise
            if df_out['signal'].loc[t_s] == cfg.train_signal_type:
                meta_label = 1.0 if max_rise >= 0.2 else 0.0
            else:
                meta_label = np.nan

            label_values.append(label_value)
            meta_labels.append(meta_label)

        locs = [df_out.index.get_loc(t) for t in signal_times]
        df_out.iloc[locs, df_out.columns.get_loc(cfg.label_value_col)] = label_values
        df_out.iloc[locs, df_out.columns.get_loc(cfg.meta_label_col)] = meta_labels

        return df_out, {"labeler": self.name}
# ============================================================
# 7. 标注器注册表
# ============================================================

LABELER_REGISTRY = {
    "erdf_evt": ERDFEVTLabeler,
    "fixed_extrema": FixedExtremaLabeler,
    "future_return": FutureReturnLabeler,
    "triple_barrier_meta": TripleBarrierMetaLabeler,
    "up20pct":MyUp20PercentLabeler
}


def build_labeler(label_method):
    """
    根据字符串名称创建标注器。

    示例
    ----
    labeler = build_labeler("erdf_evt")
    df, info = labeler.transform(df, cfg)
    """
    if label_method not in LABELER_REGISTRY:
        raise ValueError(
            f"未知标注方法: {label_method}. "
            f"可选方法: {list(LABELER_REGISTRY.keys())}"
        )

    return LABELER_REGISTRY[label_method]()