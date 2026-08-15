# SQLPilot：Qwen2.5-Coder-3B 的 Text-to-SQL QLoRA

SQLPilot 是一个面向 Spider 1.0 的端到端 Text-to-SQL 实验项目，覆盖数据校验、四级输入增强、4-bit QLoRA 训练、断点续跑推理，以及官方 EM、EX、TSA 评测。基础模型固定为 `Qwen/Qwen2.5-Coder-3B-Instruct` 的指定 commit，正式实验在单张 NVIDIA V100 上完成。

## 最终结果

所有结果均在完整 Spider 官方 dev（1,034 条）上，由固定 commit 的 Spider 与 Test Suite 官方脚本计算。S4 Schema Link 获得最佳综合结果：**EM 65.09%、EX 72.73%、TSA 70.12%**。

| 实验 | 输入策略 | EM (%) | EX (%) | TSA (%) | 格式有效率 (%) |
|---|---|---:|---:|---:|---:|
| Base-S4 | S4 Prompt，未加载 Adapter | 29.30 | 34.04 | 54.84 | 99.52 |
| S1 | DDL | 59.67 | 70.89 | 66.54 | 100.00 |
| S2 | 结构化 Schema | 62.77 | 71.18 | 67.02 | 100.00 |
| S3 | 结构化 Schema + 示例值 | 64.02 | 72.73 | 69.44 | 100.00 |
| S4 | Schema Link | **65.09** | **72.73** | **70.12** | 99.90 |

相对同 Prompt 的零样本 Base-S4，最终 S4 分别提升 **35.79、38.69、15.28 个百分点**。S1→S4 的 EM 与 TSA 持续上升，EX 在 S3 与 S4 持平。脱敏后的机器可读指标、哈希与复现边界见 [results_summary](results_summary/README.md)。

> Base-S4 只构成 S4 的严格同 Prompt 零样本对照；项目没有为 S1–S3 分别运行同格式 baseline，因此不能把各阶段相对 Base-S4 的差异完全解释为单一输入策略的因果增益。

## 项目内容

- 对 Spider 训练集、dev、Schema 与 SQLite 数据库做一致性检查，并按数据库隔离内部验证集；
- 构造 DDL、结构化 Schema、示例值和 Schema Link 四组训练输入；
- 使用 4-bit NF4、completion-only loss 和 LoRA 完成 S1–S4 消融训练；
- 固定模型 revision、随机种子、精度策略和生成参数；
- 推理结果逐条落盘，支持中断后安全续跑；
- 封装固定版本的 Spider EM/EX 与 Test Suite TSA 官方评估器；
- 记录输入、Gold、官方脚本和预测文件的 SHA-256，便于复核。

## 实验设置

| 项目 | 设置 |
|---|---|
| 基础模型 | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| 模型 revision | `488639f1ff808d1d3d0ba301aef8c11461451ec5` |
| 训练 / 内部验证 / 官方 dev | 7,939 / 717 / 1,034 |
| 量化 | 4-bit NF4，double quant |
| LoRA | `r=16`，`alpha=32`，`dropout=0.05` |
| 上下文 | 4,096 tokens，拒绝截断 |
| 训练 | 3 epochs，batch 1，gradient accumulation 16 |
| 优化器 | paged AdamW 8-bit，学习率 `2e-4`，cosine schedule |
| 推理 | greedy，`num_beams=1`，`max_new_tokens=512` |
| 正式硬件 | NVIDIA V100，FP16，BF16/TF32 关闭 |
| 随机种子 | 42 |

Token 预检没有发现超过 4,096-token 上限的样本：

| 数据 | train P99 | train 最大值 | 截断数 |
|---|---:|---:|---:|
| S1 DDL | 2,106 | 2,249 | 0 |
| S2 结构化 | 2,398 | 2,541 | 0 |
| S3 示例值 | 3,429 | 3,572 | 0 |
| S4 Schema Link | 3,463 | 3,638 | 0 |

## 目录结构

```text
configs/                  # 通用、V100 与现代 GPU 配置
scripts/                  # 数据、训练、推理、评估和环境冻结入口
src/sqlpilot/             # 核心实现
tests/                    # 数据、配置、训练与评测测试
results_summary/          # 可公开的脱敏结果证据
EVALUATION.md             # 官方评测闭环
HARDWARE_PROFILES.md      # 不同 GPU 的环境与精度说明
```

`data/`、`checkpoints/`、`logs/`、完整 `results/`、TSA 数据库和官方评估器副本不会提交到 Git；它们可由数据准备、训练和评测脚本生成，或应放在独立制品存储中。

## 安装

推荐在 Linux GPU 服务器使用 Python 3.11：

```bash
conda create -n sqlpilot python=3.11 -y
conda activate sqlpilot
pip install -r requirements.txt
pip install -e .
python scripts/validate_training_stack.py
python -m pytest -q
```

锁定的训练栈及 V100、RTX 4090/GB10 对应命令见 [HARDWARE_PROFILES.md](HARDWARE_PROFILES.md)。无 Hugging Face 网络时，需要提前复制上述模型 commit 的完整缓存，并设置 `HF_HUB_OFFLINE=1`。

## 数据准备

本仓库不重新分发 Spider 或 TSA 数据库。取得 Spider 1.0 数据后，将其放到配置指定的 `data/raw/spider_data/`，然后运行：

```bash
python scripts/check_data.py
python scripts/prepare_training_data.py --skip-data-check
python scripts/prepare_evaluation_data.py
```

数据流水线会验证训练与 dev 数据库隔离、Gold SQL、Schema/SQLite 对照、样本数量和内容哈希。生成的 1,034 条官方 dev 只用于最终评估，不会传给训练器。

## 训练

每组正式训练前先执行预检：

```bash
python scripts/train.py --config configs/v100/experiments/s1.yaml --preflight-only
python scripts/train.py --config configs/v100/experiments/s1.yaml

python scripts/train.py --config configs/v100/experiments/s2.yaml
python scripts/train.py --config configs/v100/experiments/s3.yaml
python scripts/train.py --config configs/v100/experiments/s4.yaml
```

预检会核对数据哈希、固定模型 revision、chat template、EOS、completion-only loss、Token 长度和 dev 隔离。训练器保留内部验证损失最低的 checkpoint，并在结束时回载对应 Adapter。

## 推理与官方评测

以 V100 上的 S4 为例：

```bash
python -u scripts/predict.py \
  --config configs/v100/evaluation/s4_adapter.yaml \
  --resume

python scripts/evaluate_predictions.py \
  --predictions results/evaluation/v100/s4_adapter/predictions.jsonl \
  --output-dir results/evaluation/v100/s4_adapter/official_metrics \
  --test-suite-db tools/official_evaluation/test_suite_databases/database
```

官方评估器的安装、离线 NLTK 资源和完整 baseline/Adapter 命令见 [EVALUATION.md](EVALUATION.md)。

## 结果与制品策略

公开仓库仅保留聚合指标、哈希、配置和复现脚本：

- `results_summary/metrics.csv`：适合直接查看和制图的结果表；
- `results_summary/official_metrics.json`：实验输入哈希、官方脚本版本及评测策略；
- `configs/`：所有正式实验配置；
- `scripts/`、`src/`、`tests/`：完整训练和评测实现。

完整日志、逐条模型输出与 Adapter checkpoint 可能体积较大，且环境文件可能含本机绝对路径，因此默认不进入 Git。需要公开时，应先脱敏并通过 GitHub Release、对象存储或模型仓库单独发布，同时记录其 SHA-256。

## 局限

- 正式训练只运行了随机种子 42，当前结果不包含多种子均值与方差；
- 只完成了 S4 Prompt 的零样本 baseline，S1–S3 没有各自的严格同 Prompt baseline；
- 尚未加入执行反馈修复、跨数据集鲁棒性评测或系统性错误分类；
- Spider 与基础模型受各自许可证约束，使用者应自行核对数据与模型的授权条件。
