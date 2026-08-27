# SQLPilot 官方评估闭环

本流程固定使用 Spider 1.0 官方 dev 的 1,034 条样本。零样本 baseline 与对应 Adapter 使用相同 Prompt、模型 commit、量化、精度和贪心解码配置；唯一变量是是否加载 Adapter。

已复现的五组正式结果见 [results_summary](results_summary/README.md)。其中 S4 达到 EM 65.09%、EX 72.73%、TSA 70.12%。

## 1. 生成并核对官方 dev

```bash
python scripts/prepare_evaluation_data.py
```

脚本会核对 S1–S4 的 `sample_id`、数据库和 Gold SQL，并生成：

```text
data/evaluation/spider_dev/
├── gold.sql
├── samples.jsonl
└── manifest.json
```

Gold 只在生成结束后用于评估；`scripts/predict.py` 只把记录中的 `prompt` 传给模型。无 Hugging Face 网络时，可以设置 `HF_HUB_OFFLINE=1`，但本地必须已有配置所固定 commit 的完整模型缓存。

## 2. 准备官方评估器和 TSA 数据库

脚本固定以下官方仓库 commit：

- `taoyds/spider`：`b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c`
- `taoyds/test-suite-sql-eval`：`e97acc546ecbee8fa27fa8dbf025ef61493a876c`

先下载官方 `testsuitedatabases.zip`，再执行：

```bash
python scripts/setup_official_evaluators.py \
  --test-suite-archive tools/official_evaluation/testsuitedatabases.zip
```

官方 TSA 下载页：

```text
https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view
```

脚本会检查 ZIP 路径越界，忽略 macOS 生成的 `__MACOSX`/`._*` 元数据，并自动定位真实的数据库根目录。成功时默认路径为：

```text
tools/official_evaluation/test_suite_databases/database
```

若服务器不能访问 GitHub，可以在联网机器上先运行安装脚本，再将整个 `tools/official_evaluation/` 目录复制到服务器；固定 commit 仍应保留。

### NLTK 离线资源

Spider 官方解析器依赖 NLTK。较新版本 NLTK 除 `punkt` 外还会读取 `punkt_tab`。联网安装：

```bash
python -m nltk.downloader \
  -d tools/official_evaluation/nltk_data \
  punkt punkt_tab
```

离线服务器应复制该目录，并在评估时设置：

```bash
export NLTK_DATA="$PWD/tools/official_evaluation/nltk_data"
```

这一步只使用 CPU，不需要 GPU。

## 3. 运行 baseline 与 Adapter 推理

正式跑 1,034 条前先冒烟 5 条。以下为 V100 示例；RTX 4090 把硬件参数改为 `4090`：

```bash
python -u scripts/predict.py \
  --config configs/evaluation/base_s4.yaml \
  --hardware v100 \
  --limit 5

python -u scripts/predict.py \
  --config configs/evaluation/base_s4.yaml \
  --hardware v100 \
  --resume
```

S4 Adapter：

```bash
python -u scripts/predict.py \
  --config configs/evaluation/s4_adapter.yaml \
  --hardware v100 \
  --limit 5

python -u scripts/predict.py \
  --config configs/evaluation/s4_adapter.yaml \
  --hardware v100 \
  --resume
```

S1–S3 分别使用 `s1_adapter.yaml`、`s2_adapter.yaml`、`s3_adapter.yaml`。所有正式推理固定：

```text
do_sample=false
num_beams=1
max_new_tokens=512
batch_size=1
```

预测逐条写入 `results/evaluation/<name>/predictions.jsonl`，中断后可用 `--resume` 继续。

## 4. 运行 EM、EX 与 TSA

评测本身只使用 CPU。以 S4 Adapter 为例：

```bash
env NLTK_DATA="$PWD/tools/official_evaluation/nltk_data" \
python scripts/evaluate_predictions.py \
  --predictions results/evaluation/s4_adapter/predictions.jsonl \
  --output-dir results/evaluation/s4_adapter/official_metrics \
  --test-suite-db tools/official_evaluation/test_suite_databases/database
```

输出包括：

```text
pred.sql
official_spider_em_ex.log
official_test_suite_tsa.log
metrics.json
```

`metrics.json` 记录评测输入、执行命令、EM/EX/TSA 与格式合规率。评测策略固定为：

- 模型自行预测查询值，不启用 `--plug_value`；
- 遵循官方默认，不启用 `--keep_distinct`；
- 数量和顺序必须严格对应完整 1,034 条 dev；
- 不允许手工修改预测 SQL。

## 5. 公开结果

完整 `results/` 和 `logs/` 默认不提交 Git。发布前只从每组 `official_metrics/metrics.json` 提取聚合指标和固定版本信息，去除绝对路径、运行命令中的用户名及机器环境。当前公开摘要位于：

```text
results_summary/metrics.csv
results_summary/official_metrics.json
```

原始预测、日志与 checkpoint 如需共享，应作为独立制品发布；不要直接提交官方数据库、模型权重或含本机路径的环境快照。

## 公平性约束

- baseline 与 Adapter 必须使用同一份 evaluation-only JSONL；
- 不允许把 `completion` 或官方 dev Gold 放入生成 Prompt；
- 不允许截断 Prompt；
- 不允许手工修正预测 SQL；
- 单 GPU 上不要让训练和模型推理并发运行；
- EM/EX/TSA 评测无需占用 GPU，可以在推理完成后前台或后台执行。
