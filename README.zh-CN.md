# ECG-digitizer 复现

[English](README.md) | **简体中文**

本仓库直接复现论文 *Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research* 的 ECG 图像数字化方法，基于作者的 [Open-ECG-Digitizer 开源代码](https://github.com/Ahus-AIM/Open-ECG-Digitizer) 开展训练与推理。

复现沿用原项目的分割网络与数字化流程，并在此基础上加入多卡训练、V100 / A100 适配、训练数值保护以及部分推理改进。另外使用 PTB-XL / Chapman 波形生成了新的合成训练数据，提供自己训练的分割权重。以下说明本次复现所做的修改、数据生成方式和使用方法。

## 在原项目基础上的改动

- **多卡训练**：在 `src/train.py` / `src/utils.py` 中加入 DDP、DistributedSampler、自动多卡启动和显式 `torchrun`，仅 rank 0 保存模型。
- **GPU 与数值适配**：V100 的 Muon 使用 FP32，A100 支持 BF16；loss 使用 FP32，增加梯度裁剪、非有限值保护与可配置 InstanceNorm eps。
- **自建数据**：新增页面渲染与数据生成脚本，从 PTB-XL / Chapman 波形生成三种布局、15 种网格样式的 ECG 图像及 RGB mask。仓库仅提供生成脚本和说明，不上传自建图像。
- **推理改进**：调整裁剪与信号内容边界，处理校准脉冲，并补充输出 metadata。
- **训练与推理配置**：提供单卡 / 多卡配置、两份 A100 Slurm 脚本、自定义模型保存路径，以及匹配新权重的推理配置。
- **自训练权重**：发布 `npj_newdata_weights_39.pt`，并将原项目模型与自己训练的模型分开标注。

逐文件的修改说明见 [docs/CHANGES.md](docs/CHANGES.md)，原方法与原始模型的署名见 [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)。

## 安装

需要 Python 3.12。GPU 训练需要与驱动兼容的 PyTorch CUDA 安装；本次整理时的可用环境是 Python 3.12.0、PyTorch 2.6.0+cu124，不代表历史作业保存过完整依赖锁。

```bash
git clone https://github.com/cherishleon/ECG-digitizer.git
cd ECG-digitizer
git lfs install
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 保留原项目依赖；`reports/environment_snapshot.json` 是整理时环境快照，不是保证逐位一致的训练环境。安装 PyTorch 时根据自己的 CUDA 环境选择发行版。

## 新增数据

本次复现使用 **PTB-XL 的公开 12 导联 WFDB 波形**与 **Chapman 的 12 导联 CSV 波形**，重新渲染为带网格、导联文字及增强的 ECG 页面，并同步生成分割 mask，用于本项目的分割模型训练。

主生成入口为 `generate_open_ecg_gridtype_dataset.py`，配合 `generate_open_ecg_segmentation_dataset.py`（布局与掩码）、`ecg_page_rendering.py`（页面与网格）以及 `generate_synthetic_3x4_rhythm_dataset.py`（波形读取工具）。这些脚本随仓库提供，使用自己的源数据路径运行即可。

本次复现共生成 **116,500 对 PNG 图像 / mask**，来自 32,387 个源记录：

| 项目 | 数量 |
| --- | ---: |
| train | 98,694 |
| val | 17,806 |
| test | 0 |
| PTB-XL 衍生页 | 78,396 |
| Chapman 衍生页 | 38,104 |
| 正面页 | 97,161 |
| 额外透视页 | 19,339 |

布局包含 `4x3_plus_3`、`6x2_plus_1`、`4x3_plus_1`。这里 `4x3` 表示 4 列 × 3 行，即常说的 3×4 ECG 页面。Mask：R=网格，G=文字 / 校准脉冲 / 干扰标记，B=生理 ECG 曲线；背景由剩余概率得到。

**自建图像、mask、原始波形和逐条 manifest 均不上传 GitHub**，只提供生成程序、数据来源、配方与汇总统计。具体参数、波形格式、划分方式及局限见 [数据说明](docs/DATASET.md)。上游程序原有的固定增强资源与小型测试样例不属于本次自建数据，保留以确保程序和测试可运行。

历史生成采用记录级 85% / 15% / 0% 划分。配置里的 TEST 暂时借用 val，仅用于满足现有 loader 的接口，**没有独立测试集结果**。需要独立测试集时请重新生成，例如 `--split-ratios 0.8,0.1,0.1`，并修改 TEST 路径；这会改变数据划分和训练结果。

下载原始波形后，先做小规模生成：

```bash
python generate_open_ecg_gridtype_dataset.py --ptb-root /path/to/ptb-xl --chapman-root /path/to/chapman_csv --output-dir data/smoke_dataset --limit-ptb 2 --limit-chapman 2 --workers 2 --seed 1234
```

重建历史配方：

```bash
python generate_open_ecg_gridtype_dataset.py --ptb-root /path/to/ptb-xl --chapman-root /path/to/chapman_csv --output-dir data/npj_ecg_dataset_new --workers 24 --seed 1234 --split-ratios 0.85,0.15,0
```

启用中文报告文本时需要中文字体，可以加 `--chinese-font /path/to/font.ttf`。没有字体时可用 `--chinese-text-probability 0` 做功能检查，但不再是历史配方。Chapman 输入为 500 Hz 的 12 导联 CSV，默认单位换算除以 1000；不要直接传入未经确认格式的 WFDB 数据。

## 单卡 / 多卡训练

从仓库根目录执行。优先使用 A100 BF16 配置；40 GB 显存卡需自行降低 batch，历史作业使用 A100-SXM4-80GB。

```bash
# 仅暴露一张卡：单进程训练
CUDA_VISIBLE_DEVICES=0 python -m src.train --config src/config/unet_a100_newdata.yml

# 与历史 newdata 脚本对应：2 张 A100
torchrun --standalone --nproc-per-node=2 --module src.train --config src/config/unet_a100_newdata.yml

# v2 当前脚本：4 张 A100
torchrun --standalone --nproc-per-node=4 --module src.train --config src/config/unet_a100_v2.yml

# 自动使用所有可见 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train --config src/config/unet_a100_v2.yml
```

`DATALOADER.KWARGS.batch_size` 是**每卡 batch**，全局 batch=每卡 batch×GPU 数；当前未实现梯度累积。

| 配置 / 脚本 | GPU 数 | 每卡 batch | 全局 batch | epochs | AMP | 初始 LR | T_max |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| newdata / `run_a100_newdata.sh` | 2 | 20 | 40 | 40 | BF16 | 0.0037 | 46,500 |
| v2 / `run_a100_v2.sh` | 4 | 20 | 80 | 40 | BF16 | 0.0037 | 46,500 |

T_max 按训练 batch 计数，最低 LR=0.00037，之后恒定；跳过优化器更新的 batch 仍推进 scheduler。以 98,694 张图估算：2 卡每轮 2,467 steps，约第 18.85 轮降至最低；4 卡每轮 1,233 steps，约第 37.71 轮降至最低。因此更改卡数并不只改变速度，也改变余弦调度覆盖的 epoch 数。这两份配置不是完全等价实验。

新数据路径写作 `../../data/npj_ecg_dataset_new/train` 等，是因为现有 ECGScanDataset 的相对路径以 `src/dataset/` 为基准。也可以全部改为自己的绝对路径。

设置 `TRAIN.KWARGS.checkpoint_dir` 可选择保存路径（相对仓库根目录）：

```yaml
TRAIN:
  KWARGS:
    checkpoint_dir: 'outputs/a100_newdata'
```

仅 rank 0 保存 `weights_<epoch>.pt`，不带 DDP / compile 前缀。普通训练不使用 YAML 中的 Ray `STOPPER`，会执行设定 epochs。验证有随机裁剪，且默认 drop_last=True，因此不是全图、确定性验证。

V100 示例是 `src/config/unet.yml`，采用 FP32 forward 与 FP32 Muon。其数据路径指向旧数据目录，需要自行准备；不要直接假设每卡 batch 8 能在 FP32 下装进 32 GB 显存，建议小 batch 做试跑。不能把 V100 与 A100 直接混成同一作业，因为当前 Muon 会按 GPU 选择不同计算 dtype。

恢复训练入口为 `--resume`，读取输出目录中的 `checkpoint_last.pt`。本次提供的第 39 轮文件只有 model state_dict，可用于推理，不能等价恢复训练。现有自定义 AdamMuon 的子优化器 state 尚未完整纳入外层序列化，随机数状态也未保存；恢复功能不保证与未中断训练一致，详见 [已知限制](docs/CHANGES.md)。

## Slurm

在集群上先激活自己的环境，再提交。脚本不硬编码用户目录或项目账号：

```bash
mkdir -p logs
sbatch --account=<YOUR_ACCOUNT> --export=ALL,ECG_PYTHON=/path/to/env/bin/python run_a100_newdata.sh
sbatch --account=<YOUR_ACCOUNT> --export=ALL,ECG_PYTHON=/path/to/env/bin/python run_a100_v2.sh
```

仓库路径优先使用 `ECG_REPO_DIR`，否则读取 `SLURM_SUBMIT_DIR`；非 Slurm 调用时使用脚本所在目录。通过 `ECG_PYTHON`、`ECG_CONFIG`、`ECG_TRITON_CACHE_DIR` 可分别设置 Python、配置和缓存路径。GPU 型号名 `a100`、资源配额和日志目录按集群修改，账号只在提交时传入，不写入脚本。

## 权重与推理

本项目自训练权重只发布在新增合成数据上训练得到的 `npj_newdata_weights_39.pt`，不发布其他训练 epoch 的权重。另外保留原项目模型，并明确区分来源。

| 文件 | 用途 | 来源 |
| --- | --- | --- |
| `checkpoints/npj_newdata_weights_39.pt` | 自训练四类 ECG 页面分割 | newdata，epoch 39，验证 loss 0.0435144846 |
| `checkpoints/unet_weights_07072025.pt` | 原始四类页面分割 | 上游原始文件 |
| `checkpoints/lead_name_unet_weights_07072025.pt` | 导联文字识别 | 上游原始文件；没有重新训练 |

详见 [权重说明](checkpoints/README.md)，校验信息在 `reports/checkpoints.json`。Git LFS 下载失败时会留下文本 pointer，无法直接 torch.load。

```bash
python -m src.digitize --config src/config/inference_newdata.yml DATA.images_path=/path/to/ecg_images DATA.output_path=outputs/inference
```

自训练权重必须使用 `norm_eps: 1.0e-2`，该值不会保存在 state_dict 中。原始分割权重保持 `1.0e-5`。公开版 `inference_newdata.yml` 已配置正确值；识别布局可通过 `lead_layouts_*.yml` 限制。输出波形单位为 μV。

针对固定版式，还提供 `inference_3x4_newdata.yml`（3×4 短导联 + 3 条长节律）和 `inference_6x2_newdata.yml`（6×2 短导联 + 1 条长节律）。配置名只描述布局，不包含本地数据来源标签。

本地完整日志记录 2×A100-SXM4-80GB、40 epochs：第 39 轮 train_loss=0.0573955938，val_loss=0.0435144846；第 40 轮 val_loss=0.0435892576。这是合成数据随机裁剪验证损失，不是独立测试成绩或波形重建精度。日志仍记录若干非有限梯度步骤被跳过，不能解释成所有步骤均稳定。逐轮指标见 `reports/training_newdata.csv`。

## 文件导航

| 路径 | 作用 |
| --- | --- |
| `src/train.py` | DDP 启动、精度控制、训练 / 验证、checkpoint |
| `src/utils.py` | DistributedSampler、余弦至恒定 LR、rank 0 绘图 |
| `src/model/unet.py` | 分割网络、可配置 norm_eps、冻结未使用层 |
| `src/optimizer/adammuon.py` | Muon + AdamW、dtype 选择、参数组 weight decay |
| `src/transform/vision.py` | 数据增强、Fourier 归一化除零保护 |
| `ecg_page_rendering.py` | 页面几何、网格与掩码合成 |
| `generate_open_ecg_gridtype_dataset.py` | 推荐数据生成入口 |
| `generate_open_ecg_segmentation_dataset.py` | gridtype 依赖的三布局渲染逻辑 |
| `generate_synthetic_3x4_rhythm_dataset.py` | 波形读取及旧渲染工具；旧六通道 mask 不可直接用于当前训练 |
| `src/model/cropper.py`、`inference_wrapper.py` | 内容保留、裁剪改进、校准脉冲处理 |
| `docs/`、`reports/` | 修改、数据、发布说明和可核查实验记录 |

## 本地检查

```bash
python -m pip install pytest
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q test
python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/smoke_ddp.py
```

测试会在忽略的 `sandbox/` 中生成临时权重。CPU/Gloo 检查不替代实际 GPU 训练和端到端精度评测；本次检查结果见 `reports/release_validation.json`。

## 复现来源、许可与引用

原方法与原始实现来自 [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer)。本仓库在其基础上进行复现与修改，保留原项目 [CC BY-SA 4.0 LICENSE](LICENSE)，新增代码按同一许可提供。数据来源保留各自授权和引用，见 [第三方说明](docs/THIRD_PARTY.md)。原始临床 JPG/XML、私有叠加图、患者记录、集群凭据和训练日志不随仓库发布。

请引用原论文：

```bibtex
@article{stenhede_digitizing_2026,
  title = {Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research},
  author = {Stenhede, Elias and Bjørnstad, Agnar Martin and Ranjbar, Arian},
  journal = {npj Digital Medicine},
  year = {2026},
  doi = {10.1038/s41746-025-02327-1}
}
```

数据生成的详细说明见 [docs/DATASET.md](docs/DATASET.md)，逐文件修改与当前限制见 [docs/CHANGES.md](docs/CHANGES.md)。
