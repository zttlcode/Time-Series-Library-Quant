使用说明
准备数据
将信号文件（signals.csv）和历史行情文件（ohlcv.csv）放在工作目录，格式严格按照题目要求。

运行脚本
执行以上代码，会自动完成：

技术指标计算

极值点检测与 GPD 拟合（生成标签 
D
,
ξ
label
D,ξ 
label
​
 ）

对比学习正负样本构建

CTAML 模型训练（BiLSTM 编码器 + TAA 可选 + 多任务输出 + InfoNCE）

模型保存为 best_ctaml.pth

后续实验

消融实验：通过修改 Config 中的开关（use_taa, lambda_cl, 或用单点回归替代分位数）生成变体。

回测：调用 apply_filter_and_trade 评估绩效，计算夏普比率、最大回撤等指标。

依赖安装

bash
pip install pandas numpy torch scipy scikit-learn matplotlib tqdm
代码严格按照论文框架设计，所有模块名称与论文一致，可直接用于论文实验部分。