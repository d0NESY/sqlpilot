# 官方结果摘要

这里保存适合公开提交的聚合结果与复现证据。数字来自本地完整 `results/evaluation/` 中五组通过状态的 `official_metrics/metrics.json`，但已移除用户名、服务器绝对路径、运行命令和官方评估器的运行时副本。核对脚本也兼容早期保存在 `v100/` 子目录中的归档。

## Spider dev 结果

| 实验 | EM (%) | EX (%) | TSA (%) | 格式有效率 (%) |
|---|---:|---:|---:|---:|
| Base-S4 | 29.30 | 34.04 | 54.84 | 99.52 |
| S1 DDL | 59.67 | 70.89 | 66.54 | 100.00 |
| S2 Structured | 62.77 | 71.18 | 67.02 | 100.00 |
| S3 Values | 64.02 | 72.73 | 69.44 | 100.00 |
| S4 Schema Link | **65.09** | **72.73** | **70.12** | 99.90 |

S4 相对同 Prompt 的 Base-S4 提升 35.79 EM、38.69 EX、15.28 TSA 个百分点。S1→S4 的 EM 与 TSA 单调上升，EX 在 S3 与 S4 持平。

## 文件说明

- `metrics.csv`：便于表格软件、绘图脚本和简历材料直接使用；
- `official_metrics.json`：包含指标、输入哈希、模型 revision、官方仓库 commit 与官方脚本哈希。

若本地仍保留完整原始结果，可在提交前核对摘要没有抄录错误：

```bash
python scripts/verify_results_summary.py
```

所有实验使用完整 1,034 条 Spider 1.0 dev。`status` 均为 `passed`，Gold SQL、`tables.json` 和官方评估脚本的哈希在五次运行中一致。模型自行预测查询值，未启用 `--plug_value`；评估遵循默认 distinct 策略，未启用 `--keep_distinct`。

## 未公开的本地制品

下列内容由 `.gitignore` 排除：完整逐条预测、原始日志、运行环境快照、官方评估器运行时副本、Spider/TSA 数据库和 Adapter checkpoint。它们适合本地留档或单独发布为带 SHA-256 的 Release/模型制品，不适合直接放入代码仓库。

## 解释边界

- 正式结果只有随机种子 42，没有多种子均值和方差；
- Base-S4 是 S4 的严格同 Prompt 对照，不是 S1–S3 各自的匹配 baseline；
- 当前数据支持“完整训练评估闭环”和“输入增强消融趋势”的结论，但不足以声称跨数据集泛化或统计显著性。
