# Time-Series-Library-Quant（TSLQ）

> 🧠 用深度学习做 A 股量化交易：用时间序列分类模型判断交易信号质量，支持全自动实盘交易。

## 这个项目是做什么的？

一句话：**你有一个交易策略（比如 MACD 金叉买入），但这个策略会产生很多假信号。TSLQ 用深度学习模型来判断每个信号是"真信号"还是"假信号"，只做高质量的交易，从而提高胜率。**

它本质上是一个**交易信号过滤器**。你的策略给出买卖信号 → TSLQ 用时间序列深度学习模型（TimesNet、Transformer、LSTM 等）判断这个信号是否可靠 → 只执行可靠的信号。

---

## 目录

- [快速开始](#快速开始)
- [核心理念：为什么这样设计？](#核心理念为什么这样设计)
- [项目结构一览](#项目结构一览)
- [功能一：训练模型与批量回测](#功能一训练模型与批量回测) ← `main.py`
- [功能二：Windows 实盘自动交易](#功能二windows-实盘自动交易) ← `Run_prd_win.py`
- [功能三：Linux 实盘自动交易](#功能三linux-实盘自动交易) ← `Run_prd_linux.py`
- [功能四：CTAML 元标签基准实验](#功能四ctaml-元标签基准实验) ← `run_meta_labeling_benchmark.py`
- [功能五：自适应止损实验](#功能五自适应止损实验) ← `adaptive_stop_loss_experiment.py`
- [功能六：模型形状调试](#功能六模型形状调试) ← `tool.py`
- [配置说明](#配置说明)
- [如何扩展](#如何扩展)

---

## 快速开始

### 环境要求

- Python 3.9+
- CUDA（可选，有 GPU 训练更快）

### 安装

```bash
pip install -r requirements.txt
```

### 跑通第一个例子

1. **配置数据路径**：编辑 `Configs/config.ini`，把路径改成你自己的
2. **训练一个模型**：修改 `main.py` 底部的参数，然后运行：

```bash
python main.py
```

就这么简单。下面我们详细解释每一步。

---

## 核心理念：为什么这样设计？

### 问题：交易信号太多，真假难辨

任何技术指标策略（MACD、布林带、均线交叉……）都会产生大量交易信号，但其中很多是亏钱的。如果能事先知道哪些信号靠谱、哪些不靠谱，就能大幅提高收益。

### 方案：把信号判断变成"看图分类"问题

当策略发出一个买入信号时，我们不看这一个时间点，而是看**这个信号发生前的一段行情**（比如过去 160 根 K 线）。就像老股民看 K 线图判断形势一样，模型"看"这段行情图，判断这个信号能不能赚钱。

这就变成了一个**时间序列分类问题**：
- 输入：信号发生前 160 根 K 线的多维特征（价格、均线、MACD、RSI、成交量……）
- 输出：这个信号是"好信号"还是"坏信号"

<p align="center">
  <img src="https://raw.githubusercontent.com/thuml/Time-Series-Library/main/fig/timesnet.png" alt="时间序列分类" width="600"/>
</p>

### 数据格式：为什么用 `.ts` 文件？

本项目基于 [Time-Series-Library](https://github.com/thuml/Time-Series-Library)，这是一个学术界广泛使用的时间序列深度学习库。它使用 sktime 标准的 `.ts` 格式存储分类数据：

```text
@problemName A_d_fuzzy_ma_feature_fuzzy_ma_TrueU_label2_160S
@timeStamps false
@missing false
@univariate false
@dimensions 9
@equalLength true
@classLabel true 1 2 3 4
@data
123.4,45.6,0.03,0.01,...:234.5,56.7,0.02,-0.01,...:...:4
```

每行是一个样本，冒号分隔不同特征维度，最后一个冒号后面是类别标签。每个维度内部是逗号分隔的 160 个时间步数据。

### 两层架构

项目是"嫁接"出来的——在原版 Time-Series-Library 之上加了一层量化交易框架：

```
┌─────────────────────────────────────────────────────────┐
│  量化交易层（SQ* 模块）                                    │
│  · 数据准备：从 OHLCV bar 数据计算技术指标，生成 .ts 文件    │
│  · 策略管理：tea_radical_nature（2分类）、fuzzy_ma（4分类） │
│  · 实盘运行：定时任务 → 读取实时信号 → 模型推理 → 发通知     │
├─────────────────────────────────────────────────────────┤
│  深度学习层（原 Time-Series-Library）                      │
│  · 模型：TimesNet, Transformer, DLinear, LSTM, CNN 等    │
│  · 训练/验证/测试：标准 PyTorch 流程                       │
│  · 数据加载：从 .ts 文件读取，支持变长序列                    │
└─────────────────────────────────────────────────────────┘
```

---

## 项目结构一览

```
Time-Series-Library-Quant/
│
├── main.py                        # 🔥 入口一：训练/回测
├── Configs/
│   └── config.ini                 # 数据路径配置
│
├── models/                        # 深度学习模型
│   ├── TimesNet.py                # 时序二维变化建模（推荐）
│   ├── Autoformer.py              # 自相关 Transformer
│   ├── Transformer.py             # 标准 Transformer
│   ├── DLinear.py                 # 简单线性模型（强基线）
│   ├── Informer.py                # 高效长序列 Transformer
│   ├── PatchTST.py                # Patch 时序 Transformer
│   ├── iTransformer.py            # 倒置 Transformer
│   ├── Nonstationary_Transformer.py
│   ├── ClassCNN.py                # 1D CNN 分类器
│   ├── ClassLSTM.py               # LSTM 分类器
│   ├── CNN_LSTM.py                # CNN+LSTM 混合
│   └── XGB.py                     # XGBoost 封装
│
├── exp/                           # 实验流程
│   ├── exp_basic.py               # 模型注册表 + 设备管理
│   └── exp_classification.py      # 分类任务：训练/验证/测试循环
│
├── data_provider/                 # 数据加载
│   ├── data_loader.py             # UEALoader：读取 .ts 文件
│   ├── data_factory.py            # 数据工厂：路由到正确的 loader
│   └── uea.py                     # 变长序列 collate 函数
│
├── SQData/                        # 量化数据层
│   ├── Asset.py                   # 资产抽象（Stock/Index/ETF/Crypto）
│   ├── Bar.py                     # OHLCV bar 数据实体
│   ├── Position.py                # 仓位/订单管理
│   ├── Indicator.py               # 技术指标计算
│   └── Identify_market_types_helper.py  # 指标库（EMA/MACD/KDJ/Boll/ATR/RSI/ADX/OBV）
│
├── SQModel/
│   └── Dataset.py                 # 🔥 核心：特征工程 + 样本均衡 + .ts 文件生成
│
├── SQTool/
│   ├── Tools.py                   # 配置读写 + 交易日历
│   ├── Message.py                 # QQ 邮件通知
│   └── tool.py                    # 🔥 入口六：模型形状调试
│
├── SQRuns/
│   ├── run_quant.py               # 训练/推理编排器
│   ├── run_live.py                # 实盘 CSV→ts 转换 + 信号处理
│   ├── Run_prd_win.py             # 🔥 入口二：Windows 实盘
│   └── Run_prd_linux.py           # 🔥 入口三：Linux 实盘
│
├── CTAML/
│   ├── run_meta_labeling_benchmark.py  # 🔥 入口四：元标签实验
│   ├── label_methods.py           # 可插拔标注方法（ERDF-EVT/Triple Barrier/未来收益…）
│   └── label_methods_readme.md    # 标注方法扩展教程
│
├── stop_loss/
│   └── adaptive_stop_loss_experiment.py  # 🔥 入口五：自适应止损实验
│
├── layers/                        # 模型底层组件
│   ├── AutoCorrelation.py         # 自相关机制
│   ├── SelfAttention_Family.py    # 自注意力
│   ├── Embed.py                   # 数据嵌入
│   ├── Conv_Blocks.py             # 卷积块
│   ├── Autoformer_EncDec.py       # Autoformer 编解码器
│   ├── Transformer_EncDec.py      # Transformer 编解码器
│   └── StandardNorm.py            # 标准化
│
└── utils/                         # 工具函数
    ├── tools.py                   # 早停/学习率调整/准确率计算
    ├── timefeatures.py            # 时间特征编码
    ├── masking.py                 # 掩码
    ├── augmentation.py            # 数据增强
    └── dtw.py                     # DTW 距离
```

---

## 功能一：训练模型与批量回测

**入口文件**：`main.py`

这是最常用的功能。你在这里配置策略、特征、模型，然后运行训练或推理。

### 两种运行模式

#### 模式 A：单次训练（默认，每日使用）

打开 `main.py`，底部有这样一段代码：

```python
if __name__ == '__main__':
    name = 'A_d'                      # 市场_级别
    time_point_step = '160'           # 回溯时间步
    handle_uneven_samples = 'True'    # 是否均衡样本
    strategy_name = 'fuzzy_ma'        # 策略名：fuzzy_ma 或 tea_radical_nature
    feature_plan_name = 'feature_fuzzy_ma'  # 特征方案
    label_name = '_label2'            # 标签列
    model_name = 'DLinear'            # 模型：TimesNet/DLinear/ClassCNN/ClassLSTM…
    classification = 4                # 分类数：2 或 4
    classification_direction = 'buy'  # 只做 buy 侧

    run_quant.train(...)      # 训练 + 测试
    # run_quant.inference(...)  # 仅推理（取消注释用这个）
```

你只需要修改这些变量，然后 `python main.py`。

#### 关于两个策略

| 策略名 | 分类数 | 含义 | 特征方案 |
|--------|--------|------|----------|
| `tea_radical_nature` | 2 分类 | 买入信号：1=有效 / 2=无效；卖出：3=有效 / 4=无效 | `feature_tea_radical_nature` |
| `fuzzy_ma` | 4 分类 | 1=有效买 / 2=无效买 / 3=有效卖 / 4=无效卖 | `feature_fuzzy_ma` |

**2 分类关注 PR-AUC**：PR-AUC 提升且 balanced_accuracy 不降 → 特征有价值。
**4 分类关注 macro-F1**：macro-F1 上升且关键类别 recall 上升 → 特征有价值。

#### 模式 B：批量特征搜索（特征工程阶段使用）

`main.py` 中被注释掉的大段代码是批量特征搜索模式。当你在寻找最佳特征组合时，它会：

1. 自动生成所有特征组合（基础特征 + 排列组合候选特征）
2. 为每个特征组合生成训练集和测试集
3. 逐个训练模型
4. 结果写入 `results/` 目录的 `result_classification.txt`

特征生成规则在 `SQModel/Dataset.py` 中定义：
- `BASE_FEATURES`：18 个基础技术指标（各周期 EMA、MACD、ADX、ATR、布林带、RSI、OBV、量价）
- `DISTANCE_CANDIDATES`：13 个距离型候选（价格到均线的距离、成交量的 z-score 等）
- `INTERACTION_CANDIDATES`：7 个交互型候选（涨跌幅 × 成交量异常度等）
- `register_combo_plans()` 自动生成排列组合

### 训练过程发生了什么？

```
main.py 
  ↓ 调用
run_quant.train()                        # SQRuns/run_quant.py
  ↓ 组装参数，调用
Exp_Classification.train()               # exp/exp_classification.py
  ↓ 
① 从 .ts 文件加载数据（UEALoader）
② 构建模型（根据 model_dict 注册表）
③ 训练循环（RAdam 优化器 + CrossEntropyLoss + EarlyStopping）
④ 每个 epoch 在测试集上计算 accuracy / F1 / AUC / PR-AUC
⑤ 保存最佳 checkpoint 到 checkpoints/
⑥ 测试结果写入 results/
```

### 如何选择模型？

模型在 `exp/exp_basic.py` 的 `model_dict` 中注册。当前可用：

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **TimesNet** | 将 1D 时间序列转成 2D，用卷积捕捉周期模式 | 推荐首选，效果最稳定 |
| **DLinear** | 极简线性模型，但经常出人意料地好 | 强基线，训练快 |
| **Transformer** | 标准注意力机制 | 长序列依赖 |
| **Autoformer** | 自相关机制替代注意力 | 周期性强的数据 |
| **Informer** | 稀疏注意力，处理长序列 | 大数据量 |
| **iTransformer** | 倒置结构，在变量维度做注意力 | 多变量场景 |
| **PatchTST** | 把序列切成 patch 再处理 | 局部模式捕捉 |
| **Nonstationary_Transformer** | 处理非平稳序列 | 趋势明显的行情 |
| **ClassCNN** | 1D CNN | 局部特征，训练快 |
| **ClassLSTM** | LSTM 序列建模 | 顺序依赖 |
| **CNN_LSTM** | CNN 提取局部 + LSTM 建模全局 | 混合方案 |
| **XGB** | XGBoost | 表格数据强基线 |

### 重要参数约束

- `enc_in`、`dec_in`、`n_heads` **三个值必须相等**，且等于你的特征数量
- `seq_len`（时间步长）**必须 ≥ 8**
- 默认 `seq_len = 160`，即用过去 160 根 K 线来判断信号质量

---

## 功能二：Windows 实盘自动交易

**入口文件**：`SQRuns/Run_prd_win.py`

这个功能让整个系统**每天自动运行**，无需人工干预。

### 一天的工作流程

```
交易日 16:00（收盘后）
  │
  ├─ ① 运行 fuzzy_ma 策略：
  │    · 读取当天产生的交易信号 CSV
  │    · 对每个信号，取过去 160 根 K 线数据
  │    · 计算技术指标 → 转成 .ts 文件
  │    · 加载训练好的模型 → 推理 → 输出预测类别和概率
  │    · 发送 QQ 邮件通知（只通知预测正确的信号）
  │    · 清理临时文件
  │    · 删除未命中预测的持仓
  │
  ├─ 等待 30 秒
  │
  ├─ ② 运行 tea_radical_nature 策略：
  │    · 同上流程
  │
  └─ 等待到第二天早上 7:00，重新判断是否交易日
```

### 如何启动

```bash
python SQRuns/Run_prd_win.py
```

### 如果只想手动运行一次（不启动定时循环）

`main()` 函数里有 `# run_prd()` 被注释掉了，当前直接调用 `run("fuzzy_ma")` 和 `run("tea_radical_nature")`，适合手动测试。

要启动持续运行的定时任务，取消注释 `run_prd()` 即可。

### 实盘数据流详解

```
策略系统产生交易信号
  ↓ 写入 CSV 文件
live_to_ts/ 目录（如 fuzzy_ma_2024-01-15.csv）
  ↓
run_nature_prepare_dataset()           # run_live.py
  · 读取 CSV → 知道信号的时间、价格、方向
  · 读取该股票的实时 bar 数据
  · 计算全部技术指标
  · 截取最近 160 根 K 线
  · 写成 .ts 文件（模拟训练集格式）
  ↓
inference_live()                       # run_quant.py
  · 加载 checkpoint 模型
  · 对 .ts 文件推理
  · 输出：_prd_result.csv（预测类别）+ _prd_prob.csv（各类别概率）
  ↓
run_live_get_pred()                    # run_live.py
  · 比较预测类别 vs 真实类别（由策略标注系统预先标注）
  · 预测正确 → 保留持仓 + 发 QQ 通知
  · 预测错误 → 删除持仓
  · 清理所有临时文件
```

---

## 功能三：Linux 实盘自动交易

**入口文件**：`SQRuns/Run_prd_linux.py`

与 Windows 版本完全相同的逻辑，区别在于：

- 配置文件自动切换为 `config_prd.ini`（路径 `/home/z/data/github/Time-Series-Library-Quant/Configs/config_prd.ini`）
- 数据路径使用 Linux 风格（`/home/RobotMeQ_Dataset/...`）
- Docker 容器中运行

```bash
python SQRuns/Run_prd_linux.py
```

### 设计说明：跨平台路径自动切换

`SQTool/Tools.py` 中的 `read_config()` 会自动检测操作系统：
```python
if 'windows' in platform.platform().lower():
    path = "D:\\github\\Time-Series-Library-Quant\\Configs\\config.ini"
else:
    path = "/home/z/data/github/Time-Series-Library-Quant/Configs/config_prd.ini"
```

所以你**不需要手动改路径**，系统自动处理。

---

## 功能四：CTAML 元标签基准实验

**入口文件**：`CTAML/run_meta_labeling_benchmark.py`

这是一个**独立的研究模块**，用于对比不同"信号标注方法"的效果。

### 背景：什么是"标注"？

在训练信号过滤模型之前，你需要给历史信号打标签——哪些信号赚了钱（好信号），哪些亏了钱（坏信号）。**不同的标注方法会导致完全不同的过滤效果**。

CTAML 框架让你可以**插拔式地切换标注方法**，统一评估哪种标注方法最好。

### 五种标注方法

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| **ERDF-EVT** | 极端收益事件 + 广义帕累托分布尾部建模，衡量信号到未来结构性极值的时空距离 | 追求"买在低点"的最优入场 |
| **Triple Barrier** | 设置止盈/止损/时间三重边界，看先触发哪个 | 经典的量化标注方式 |
| **未来收益** | 未来 N 根 K 线后的涨跌幅 | 最简单的对照基线 |
| **固定极值** | 在固定窗口内找局部高低点 | 轻量级替代 ERDF |
| **自定义：60bar 涨 20%** | 未来 60 根 K 线内最高价涨幅 ≥ 20% | 特定策略定制 |

所有标注方法都输出两个标准列：
- `label_value`：连续标签（如距离值、收益率），用于诊断
- `meta_label`：二分类标签（1 = 好信号，0 = 坏信号），模型训练目标

### 运行

```bash
python CTAML/run_meta_labeling_benchmark.py
```

在 `Config` 类中修改 `label_method` 即可切换标注方法：
```python
label_method = 'triple_barrier_meta'  # 改成 'erdf_evt' 试试
```

### 如何自定义标注方法？

参考 `CTAML/label_methods_readme.md`，5 步即可：
1. 写一个继承 `BaseLabeler` 的类
2. 实现 `transform(df, cfg)` 方法，输出 `label_value` + `meta_label`
3. 确保只在信号行赋值，其他行留 NaN
4. 在 `LABELER_REGISTRY` 注册
5. 在 Config 中设置 `label_method = '你的方法名'`

---

## 功能五：自适应止损实验

**入口文件**：`stop_loss/adaptive_stop_loss_experiment.py`

这是一个**独立的研究模块**，探索如何用深度学习动态设定止损位。

### 核心思路

固定止损（如 -5% 无条件止损）不够灵活。这个模块用三种深度学习方法联合建模尾部风险：

1. **条件分位数预测**：预测未来最大不利偏移（Max Adverse Excursion, MAE）
2. **EVT 尾部建模**：用极值理论对极端损失做更精细的建模
3. **神经生存分析**：预测持仓存活的概率曲线

三种方法的结果融合成一个**自适应止损距离**——市场波动大时止损放宽，市场平静时止损收紧。

### 运行

```bash
python stop_loss/adaptive_stop_loss_experiment.py
```

### 与主项目的关系

这个模块**完全独立**，有自己的 `Config` 类、数据加载、模型训练和回测逻辑。它与主分类模型并行运行，不影响现有功能。

---

## 功能六：模型形状调试

**入口文件**：`SQTool/tool.py`

这是一个开发工具，用于检查自制模型（ClassCNN、ClassLSTM、CNN_LSTM）的张量形状是否正确。

```bash
python SQTool/tool.py
```

当你修改模型结构时，用这个快速检查各层输出的 shape 是否符合预期。

---

## 配置说明

`Configs/config.ini` 是全局配置，所有数据路径都在这里：

```ini
[SQData]
# 回测 bar 数据目录（含各股票的 OHLCV CSV）
backtest_bar = D:\github\RobotMeQ_Dataset\QuantData\backTest\

# 交易点（信号）目录，每个策略一个文件夹
trade_point_backtest_tea_radical_nature = D:\github\RobotMeQ_Dataset\QuantData\trade_point_backtest_tea_radical_nature\
trade_point_backtest_fuzzy_ma = D:\github\RobotMeQ_Dataset\QuantData\trade_point_backtest_fuzzy_ma\

# 实盘 bar 数据
live_bar = D:\github\RobotMeQ_Dataset\QuantData\live\

# 实盘信号 csv
live_to_ts = D:\github\RobotMeQ_Dataset\QuantData\live_to_ts\

# 模型推理结果输出
inference_live = D:\github\RobotMeQ_Dataset\QuantData\inference_live\

[SQT]
# 股票代码列表
asset_code = D:\github\RobotMeQ_Dataset\QuantData\asset_code\
# 数据根目录
quantdata_path = D:\github\RobotMeQ_Dataset\QuantData\
# QQ 邮件通知列表
mail_list_qq_d = 1031017763@qq.com

[RMQData_local]
# 通达信导出数据
tdx = D:\tools\new_tdx\T0002\export\
# 交易日历 CSV
workday_path = D:\github\RobotMeQ_Dataset\QuantData\
```

### 数据文件命名约定

回测 bar 文件：`bar_{市场}_{代码}_{级别}.csv`，如 `bar_A_000001_d.csv`
交易点文件：`{市场}_{代码}_{级别}{标签列}.csv`，如 `A_000001_d_label2.csv`

---

## 如何扩展

### 添加新模型

1. 在 `models/` 下创建新模型文件，需要有一个 `Model` 类，接收 `args` 参数
2. 在 `exp/exp_basic.py` 的 `model_dict` 中注册
3. 在 `main.py` 中将 `model_name` 设为你的模型名

### 添加新策略

1. 在 `SQModel/Dataset.py` 中：
   - 新增 `feature_plan_name` → 特征的字典映射
   - 在 `single_time_level_point_to_ts()` 中增加策略分支
   - 在 `get_point_to_ts()` 中添加策略名
2. 在 `SQRuns/run_live.py` 的 `assemble_ts_data()` 中添加相关分支
3. 在 `SQRuns/Run_prd_win.py` 和 `Run_prd_linux.py` 的 `run()` 中添加策略参数

### 添加新特征

在 `SQModel/Dataset.py` 中：
1. 在 `build_feature_bank_tea_radical_nature()` 中计算新特征
2. 将新特征名加入 `BASE_FEATURES`、`DISTANCE_CANDIDATES` 或 `INTERACTION_CANDIDATES`
3. 如果要批量搜索，调用 `register_combo_plans()` 时传入新候选池
4. 为硬编码的特征方案（`feature_tea_radical_nature`、`feature_fuzzy_ma`）在 `get_feature()` 和 `single_time_level_point_to_ts()` 中添加新字段

---

## 技术细节

### 随机种子

整个项目固定使用 `random_seed = 2021`，保证实验结果可复现。

### 样本均衡

训练数据中，不同类别的样本数量通常不均衡（比如有效信号远少于无效信号）。`Dataset.py` 中的 `_balance_indices_by_label_generic()` 会按最少类别截齐，偏向保留时间上更新的样本。

### Windows 路径长度限制

Windows 单个文件路径不能超过 260 字符。因此某些策略名在文件名中被缩短：
- `c4_oscillation_boll_nature` → `boll`
- `c4_oscillation_kdj_nature` → `kdj`

### 类别修复

原版 Time-Series-Library 在读取 `.ts` 文件时，会根据当前 fold 中实际出现的类别来编码，导致类别压缩（比如只有 1、3、4 时会把 1→0, 3→1, 4→2）。本项目在 `UEAloader._read_ts_class_names()` 中直接从文件头部 `@classLabel` 读取完整类别空间，修复了这个问题。

---

## 项目来源

本项目基于 [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)，原项目是一个全面的时间序列深度学习基准库。

在原版基础上，本项目：
- 删除了预测、插补、异常检测任务，只保留分类
- 新增了 4 个自定义分类模型（ClassCNN, ClassLSTM, CNN_LSTM, XGB）
- 新增了完整的量化交易框架（SQData/SQModel/SQTool/SQRuns）
- 新增了 CTAML 元标签标注框架
- 新增了自适应止损研究模块
