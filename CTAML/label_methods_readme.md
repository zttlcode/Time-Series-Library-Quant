好的，我们以 `FutureReturnLabeler` 为例，逐行拆解它在做什么，然后再告诉你如果你要自己设计一个新标注方法，需要遵循哪几个固定步骤。

---

## 一、`FutureReturnLabeler` 逐段讲解

### 1. 类定义和文档字符串
```python
class FutureReturnLabeler(BaseLabeler):
    name = "future_return"
```
- 继承自 `BaseLabeler`，必须有一个 `name` 属性（后续注册表用到）。
- 文档里写得很清楚：
  - `label_value` = 未来 `future_T` 根K线的简单收益率
  - `meta_label` = 如果收益率 ≥ `cfg.future_return_threshold`，则为好信号（1），否则坏信号（0）

### 2. `transform()` 方法 —— 核心逻辑
```python
def transform(self, df, cfg):
```
**入参**：
- `df`：已经完成特征计算的 DataFrame（包含 `close`, `signal` 等列）
- `cfg`：全局配置对象，我们能从里面取到 `future_T`、`future_return_threshold`、`train_signal_type`、`label_value_col`、`meta_label_col` 等参数

**出参**：
- `df_out`：在原始 df 基础上多了 `label_value` 和 `meta_label` 两列的 DataFrame
- `info`：一个字典，包含标注器名称等信息，方便诊断

#### 第一步：初始化标签列
```python
df_out = ensure_standard_label_columns(
    df,
    label_value_col=cfg.label_value_col,
    meta_label_col=cfg.meta_label_col
)
```
这行确保 `df_out` 里有 `label_value` 和 `meta_label` 两列，如果原来没有就创建并填满 `NaN`。后续我们只给信号所在行赋值，其他行保持 `NaN`。

#### 第二步：提取所有信号时间点
```python
n = len(df_out)
signal_times = df_out[df_out['signal'] != 'none'].index
```
找出所有非 `'none'` 的信号（即 `'buy'` 或 `'sell'`）所在的时间索引。

#### 第三步：遍历每个信号，计算标签
```python
for t_s in signal_times:
    idx_s = df_out.index.get_loc(t_s)          # 信号在DataFrame中的位置编号
    signal_type = df_out['signal'].loc[t_s]    # 'buy' 或 'sell'

    if idx_s + cfg.future_T >= n:              # 未来数据不够，跳过去
        label_value = np.nan
        meta_label = np.nan
    else:
        p0 = df_out['close'].iloc[idx_s]       # 信号当天收盘价
        p1 = df_out['close'].iloc[idx_s + cfg.future_T]  # 60天后的收盘价

        if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
            label_value = np.nan                # 价格异常，不标注
            meta_label = np.nan
        else:
            label_value = p1 / p0 - 1.0        # 收益率，例如 0.05 就是涨5%
            if signal_type == cfg.train_signal_type:    # 当前框架只训练 buy，所以只有 buy 才有标签
                meta_label = 1.0 if label_value >= cfg.future_return_threshold else 0.0
            else:
                meta_label = np.nan            # sell 信号暂时不管
```
核心就一句话：**看60天后的价格是涨了还是跌了**。如果涨幅≥阈值（比如0），就是好信号，否则是坏信号。

#### 第四步：把算好的标签填回 DataFrame
```python
locs = [df_out.index.get_loc(t) for t in signal_times]
df_out.iloc[locs, df_out.columns.get_loc(cfg.label_value_col)] = label_values
df_out.iloc[locs, df_out.columns.get_loc(cfg.meta_label_col)] = meta_labels
```
将刚刚算出的 `label_value` 和 `meta_label` 列表，按位置精确写入对应的行和列。

#### 第五步：返回
```python
info = {
    "labeler": self.name,
    "label_value_col": cfg.label_value_col,
    "meta_label_col": cfg.meta_label_col
}
return df_out, info
```
返回打好标签的 `df_out` 和一个简单的信息字典。

---

## 二、如果你想设计一个新的标注方法，需要做什么？

整个框架围绕 **两个标准列** 运转，所以你只需提供一个 **“给定一个信号，我告诉你它是好是坏”** 的逻辑。步骤固定如下：

### 1. 写一个继承自 `BaseLabeler` 的类
```python
class MyCustomLabeler(BaseLabeler):
    name = "my_custom_method"   # 必须唯一
```

### 2. 实现 `transform(self, df, cfg)` 方法
里面必须完成三件事：
- 调用 `ensure_standard_label_columns` 初始化标签列
- 遍历 `df[df['signal'] != 'none'].index` 的每个信号，计算出 **`label_value`（连续值）** 和 **`meta_label`（0/1）**
- 将计算结果写回 DataFrame 对应的位置

### 3. 确保输出符合规范
- `df_out` 中必须包含 `cfg.label_value_col` 和 `cfg.meta_label_col` 两列，**只在信号所在行有值，其他行可以 NaN**
- `info` 字典至少包含 `"labeler": self.name`，可以附加其他调试信息

### 4. 注册你的标注器
在模块底部的 `LABELER_REGISTRY` 字典里加一行：
```python
LABELER_REGISTRY = {
    ...
    "my_custom_method": MyCustomLabeler,
}
```

### 5. 在主配置中调用
修改 `Config` 里的 `label_method = 'my_custom_method'`，并增加对应的参数（如自定义阈值），然后整个实验流程无需任何其他改动。

---

## 三、一个极简示例：未来60根K线内最高价涨幅超过5%才算好

```python
class MyUp5PercentLabeler(BaseLabeler):
    name = "up5pct"

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
                meta_label = 1.0 if max_rise >= 0.05 else 0.0
            else:
                meta_label = np.nan

            label_values.append(label_value)
            meta_labels.append(meta_label)

        locs = [df_out.index.get_loc(t) for t in signal_times]
        df_out.iloc[locs, df_out.columns.get_loc(cfg.label_value_col)] = label_values
        df_out.iloc[locs, df_out.columns.get_loc(cfg.meta_label_col)] = meta_labels

        return df_out, {"labeler": self.name}
```

把这个类注册到 `LABELER_REGISTRY`，设置 `cfg.label_method = 'up5pct'`，你的新标注方法就无缝接入了整个训练→回测→评估流水线。

---

一句话总结：**框架只要求你回答“某个信号是不是好信号”，然后用两个标准列（连续值+二分类）来表达这个答案。剩下的训练、校准、回测完全自动适配。**