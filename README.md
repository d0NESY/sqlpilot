# SQLPilot：Qwen2.5-Coder-3B Text-to-SQL QLoRA

SQLPilot 是一个面向 Spider 1.0 的完整 Text-to-SQL 实验项目，覆盖数据校验、四级输入增强、4-bit QLoRA 训练、可恢复推理，以及官方 EM、EX、TSA 评测。正式实验使用固定 revision 的 `Qwen/Qwen2.5-Coder-3B-Instruct`，在单张 NVIDIA V100 上完成。

## 最终结果

全部结果均来自完整 Spider 官方 dev（1,034 条）和固定版本的官方评估器。

| 实验 | 输入策略 | EM (%) | EX (%) | TSA (%) | 格式有效率 (%) |
|---|---|---:|---:|---:|---:|
| Base-S4 | S4 Prompt，未加载 Adapter | 29.30 | 34.04 | 54.84 | 99.52 |
| S1 | DDL | 59.67 | 70.89 | 66.54 | 100.00 |
| S2 | 结构化 Schema | 62.77 | 71.18 | 67.02 | 100.00 |
| S3 | 结构化 Schema + 示例值 | 64.02 | 72.73 | 69.44 | 100.00 |
| S4 | Schema Link | **65.09** | **72.73** | **70.12** | 99.90 |

S4 相对同 Prompt 的零样本 Base-S4 提升 35.79 EM、38.69 EX、15.28 TSA 个百分点。机器可读指标和评估哈希见 [results_summary](results_summary/README.md)。

> Base-S4 只构成 S4 的严格同 Prompt 对照；S1–S3 没有分别运行匹配 baseline，因此不能把它们相对 Base-S4 的差异完全解释为单一输入策略的因果增益。

## 项目结构

```text
configs/
├── data/                 # 已知无效 Gold 的精确排除记录
├── experiments/          # S1-S4、过拟合和冒烟训练配置
└── evaluation/           # baseline 与 Adapter 推理配置
results_summary/          # 可公开的脱敏指标与哈希
scripts/                  # 数据、训练、推理和官方评测入口
src/sqlpilot/             # 扁平化核心模块
tests/                    # 单元与一致性测试
```

原始数据、checkpoint、完整日志、逐条预测、TSA 数据库和官方评估器副本均由 `.gitignore` 排除。

## 安装

项目只保留一个依赖文件 `requirements.txt`。V100 与 RTX 4090 仅在 PyTorch wheel 和计算精度上不同：

| GPU | PyTorch | 训练精度 | TF32 |
|---|---|---|---|
| Tesla V100 | CUDA 12.6 wheel | FP16 | 关闭 |
| RTX 4090 | CUDA 12.8 wheel | BF16 | 开启 |

```bash
conda create -n sqlpilot python=3.11 -y
conda activate sqlpilot

# V100：二选一
pip install torch==2.11.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# RTX 4090：二选一
pip install torch==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 其余依赖两种 GPU 共用
pip install -r requirements.txt
pip install -e .

python scripts/validate_training_stack.py
python -m pytest -q
```

## 数据准备

本仓库不重新分发 Spider 或 TSA 数据库。取得 Spider 1.0 后，将数据放到 `data/raw/spider_data/`，然后运行：

```bash
python scripts/check_data.py
python scripts/prepare_training_data.py --skip-data-check
python scripts/prepare_evaluation_data.py
```

流水线会验证训练/dev 数据库隔离、Gold SQL、Schema/SQLite 对照、样本数量和内容哈希。生成的 1,034 条官方 dev 只用于最终评估。

## 训练与推理

配置文件名与硬件无关。通过统一参数选择计算精度：

```bash
# V100
python scripts/train.py \
  --config configs/experiments/s4.yaml \
  --hardware v100

# RTX 4090
python scripts/train.py \
  --config configs/experiments/s4.yaml \
  --hardware 4090
```

推理使用同样的参数：

```bash
python scripts/predict.py \
  --config configs/evaluation/s4_adapter.yaml \
  --hardware v100 \
  --resume
```

也可以用一个脚本串行完成 baseline、S1–S4 预检、训练和推理：

```bash
# 参数依次为硬件与物理 GPU 编号
nohup bash scripts/run_pipeline.sh v100 2 \
  > logs/pipeline.log 2>&1 < /dev/null &

# RTX 4090 示例
nohup bash scripts/run_pipeline.sh 4090 0 \
  > logs/pipeline.log 2>&1 < /dev/null &
```

无 Hugging Face 网络时，提前复制固定模型 revision 的完整缓存；流水线默认设置 `HF_HUB_OFFLINE=1`，联网运行可设置 `SQLPILOT_OFFLINE=0`。

## 官方评测

```bash
python scripts/evaluate_predictions.py \
  --predictions results/evaluation/s4_adapter/predictions.jsonl \
  --output-dir results/evaluation/s4_adapter/official_metrics \
  --test-suite-db tools/official_evaluation/test_suite_databases/database
```

官方评估器、TSA 数据库与 NLTK 离线资源的准备方式见 [EVALUATION.md](EVALUATION.md)。模型自行预测查询值，不启用 `--plug_value`；遵循官方默认，不启用 `--keep_distinct`。

## 可复现性

- 基础模型 revision：`488639f1ff808d1d3d0ba301aef8c11461451ec5`；
- 训练 / 内部验证 / 官方 dev：7,939 / 717 / 1,034；
- 4-bit NF4、double quant、LoRA `r=16`、`alpha=32`、`dropout=0.05`；
- 4,096-token 上限，completion-only loss，拒绝截断；
- 3 epochs，batch 1，gradient accumulation 16，随机种子 42；
- 贪心解码，`num_beams=1`，`max_new_tokens=512`；
- 官方代码、Gold、Schema、数据集和预测文件均记录 SHA-256。

## 局限

- 正式训练只运行了随机种子 42，没有多种子均值与方差；
- 只完成了 S4 Prompt 的匹配零样本 baseline；
- 尚未加入执行反馈修复、跨数据集鲁棒性评测或系统性错误分类；
- Spider 与基础模型受各自许可证约束，使用者应自行核对授权条件。
