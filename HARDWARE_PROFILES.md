# SQLPilot 两套 GPU 运行配置

核心 Python 代码由两套服务器共用。硬件差异只放在 requirements、YAML 和
入口脚本中，避免复制训练算法后产生版本漂移。

| Profile | GPU | PyTorch wheel | 训练精度 | TF32 | 输出根目录 |
|---|---|---|---|---|---|
| `v100` | Tesla V100, CC 7.0 | CUDA 12.6 | FP16 | 关闭 | `checkpoints/v100`, `results/evaluation/v100` |
| `modern` | RTX 4090 / NVIDIA GB10 | CUDA 12.8 | BF16 | 开启 | `checkpoints/modern`, `results/evaluation/modern` |

## V100

```bash
pip install -r requirements-v100.txt
chmod +x scripts/run_hardware_profile.sh scripts/run_v100_gpu2.sh
nohup bash scripts/run_v100_gpu2.sh \
  > logs/run_v100_gpu2.log 2>&1 < /dev/null &
```

V100 固定使用 `bf16: false`、`fp16: true`、`tf32: false`。脚本默认选择
物理 GPU 2，可用 `GPU_ID` 覆盖。

## RTX 4090 / GB10

```bash
pip install -r requirements-modern.txt
chmod +x scripts/run_hardware_profile.sh scripts/run_modern_gpu.sh
GPU_ID=0 nohup bash scripts/run_modern_gpu.sh \
  > logs/run_modern_gpu.log 2>&1 < /dev/null &
```

Modern profile 固定使用 `bf16: true`、`fp16: false`、`tf32: true`。

## 离线模型

两个入口默认设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，并严格
检查固定 commit 的完整 Hub snapshot。如服务器能访问 Hugging Face，可在启动
前设置：

```bash
SQLPILOT_OFFLINE=0 bash scripts/run_v100_gpu2.sh
```

或：

```bash
SQLPILOT_OFFLINE=0 GPU_ID=0 bash scripts/run_modern_gpu.sh
```

## 共同保障

- 同一基础模型 commit、数据 SHA-256、QLoRA 参数与随机种子；
- 4-bit NF4、double quant、completion-only loss；
- 按最低 `eval_loss` 保留并回载最佳 checkpoint；
- S1-S4 全部预检通过后才开始正式串行实验；
- 任一步失败立即停止；
- 不允许在已有 checkpoint 的目录混跑；
- baseline 与对应 Adapter 使用同一硬件 profile 的计算精度；
- 两套输出目录隔离，禁止跨硬件断点续跑。
