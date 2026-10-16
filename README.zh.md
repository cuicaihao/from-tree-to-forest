# 从一棵树到一片森林：配套实验

[English](README.md) | 简体中文

![从一棵独立的决策树演变为相互连接的森林，象征树模型从单棵树走向集成学习。](feature.webp)

本仓库是博文[《从一棵树到一片森林：树模型的数学原理、算法演进与工程实践》](https://cuicaihao.github.io/zh/posts/2026-10-17-from-a-single-tree-to-a-forest-mathematical-principles-algorithmic-evolution-and-engineering-practices-of-tree-models/)的配套复现项目，集中保存文章使用的合成数据、数据生成脚本和实验 Notebook。

项目从两个贯穿全文的案例出发：

- 案例 A：银行交易欺诈分类，用于讲解分类树、类别不平衡、梯度与 Hessian、类别特征处理等概念；
- 案例 B：零售销量回归，用于讲解回归树、残差、切分增益、时间划分和预测评估。

这些数据完全由教学规则和固定随机种子生成，不来自真实银行或零售商，也不代表真实行业分布或模型表现。

## 仓库内容

| 文件 | 用途 |
| --- | --- |
| `case-a-fraud-teaching.csv` | S1—S8，共 8 笔特意设计的交易，用于文章中的手算 |
| `case-a-fraud.csv` | 6,000 笔合成交易，按标签分层保存训练、验证和测试划分 |
| `case-a-fraud.ipynb` | 欺诈分类实验：手算核对、类别统计示意、类别不平衡和三种提升树对照 |
| `case-b-retail-teaching.csv` | D1—D8，共 8 条特意设计的销量记录，用于文章中的手算 |
| `case-b-retail.csv` | 20 个商品、2 家门店、120 天组成的 4,800 行合成面板数据 |
| `case-b-retail.ipynb` | 零售回归实验：切分计算、直方图示意、时间划分和三种提升树对照 |
| `generate_tree_cases.py` | 确定性生成四份 CSV，或逐字节核对现有数据 |

教学小表和完整实验集承担不同任务：S1—S8、D1—D8 只服务于可手工核对的公式推导；完整数据只用于软件训练和评估。不要用八行教学数据上的增益直接解释完整测试集指标。

## 快速开始

需要先安装 [uv](https://docs.astral.sh/uv/)，然后创建 Python 3.12 项目环境并安装锁定的依赖：

```bash
uv sync
```

该命令会在本地创建 `.venv`；后续的 `uv run` 命令会自动使用这个环境。

克隆仓库后，需要执行一次以下命令启用 pre-commit hook：

```bash
uv run pre-commit install
```

提交时，hook 会对暂存的 `.ipynb` 文件运行 `nbstripout`，清除单元格输出和执行计数。若要手动检查所有已跟踪文件，可运行 `uv run pre-commit run --all-files`。

先核对四份固定 CSV 是否能由脚本原样生成：

```bash
uv run python generate_tree_cases.py --check
```

成功时应看到四个 `Verified` 结果，行数依次为 8、6,000、8 和 4,800。

如需执行欺诈分类 Notebook：

```bash
uv run jupyter nbconvert --to notebook --execute \
  --output case-a-executed.ipynb --output-dir /private/tmp \
  case-a-fraud.ipynb
```

执行零售回归 Notebook 时，将最后两处 `case-a` 改为 `case-b`：

```bash
uv run jupyter nbconvert --to notebook --execute \
  --output case-b-executed.ipynb --output-dir /private/tmp \
  case-b-retail.ipynb
```

执行结果写入 `/private/tmp`，不会覆盖原始 Notebook。Notebook 内的断言会核对对应 CSV、教学统计量、完整样本规模和数据划分边界。

## 数据设计边界

### 案例 A：欺诈分类

- 教学表包含 4 笔正常交易和 4 笔欺诈交易；
- 完整数据包含 6,000 笔交易，划分为 3,600 条训练、1,200 条验证和 1,200 条测试记录；
- 输入包括金额、距离、商户类型和商户 ID，`y` 是事后获得的标签，不能作为预测特征；
- 完整数据中的欺诈标签按设定概率随机抽取，因此单次合成实验不能用于给框架作普遍排名。

### 案例 B：零售销量回归

- 教学表中的折扣、`lag7` 和销量是为手算固定的输入；
- 完整数据以前 7 天形成历史窗口，第 7—79 天用于训练，第 80—99 天用于验证，第 100—119 天用于测试；
- `lag7` 先对销量执行 `shift(1)`，再计算七日滚动均值，避免把预测当天的销量泄漏给模型；
- 实验假设库存充足，目标 `y` 是完整可观察的销量，没有模拟缺货造成的需求截断；
- 该实验是每天收盘后预测下一天，不应将固定的时间留出误称为多轮滚动起点回测。

## 重新生成数据

`--check` 只比较重新生成的内容与当前 CSV，不会覆盖文件。若确实要重建四份数据，可以去掉该参数：

```bash
uv run python generate_tree_cases.py
```

此命令会覆盖四份 CSV。执行前请先保存或提交需要保留的修改。

## 如何阅读

建议先阅读博文中的案例背景和手算过程，再打开对应 Notebook：

1. 用教学 CSV 对照公式中的每个样本、梯度、Hessian 和叶节点统计量；
2. 用完整 CSV 检查训练、验证和测试流程；
3. 比较 XGBoost、LightGBM 与 CatBoost 时，关注输入语义、类别处理、权重设置和评估口径，而不是把一次合成实验的数字当作框架排行榜。
