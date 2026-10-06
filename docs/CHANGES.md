# 复现改动与当前限制

本项目直接基于作者的 Open-ECG-Digitizer 开源实现进行复现，下面列出在原实现基础上所做的修改。训练数据、参数和结果以本项目记录为准；文件差异使用原项目审计快照核对，而非仅使用本地 initial commit，因为该提交已包含部分修改。

## 训练和多 GPU

| 文件 | 修改及目的 |
| --- | --- |
| `src/train.py` | 自动 spawn / 显式 torchrun、LOCAL_RANK 设备绑定、NCCL process group 管理、rank 0 日志和保存、跨 rank 汇总 loss |
| `src/utils.py` | train / val / test DistributedSampler；train 每轮 set_epoch；绘图仅 rank 0 |
| `src/model/unet.py` | 冻结原 forward 不使用的最后 downsample 和 decoder_skips；保持原 state_dict key，避免 DDP unused parameter 错误 |
| `src/train.py` | AMP_DTYPE 支持 FP16 / BF16 / FP32；Dice + Focal 显式 FP32；backward / optimizer step 移至 autocast 外部 |
| `src/optimizer/adammuon.py` | V100 / CPU Newton–Schulz FP32，Ampere+ BF16；先 FP32 归一化；尊重每个参数组的 weight_decay |
| `src/train.py` | 梯度裁剪、FP64 汇总梯度范数、同步检查非有限 loss / BF16-FP32 梯度，避免一个 rank 单独跳过 backward |
| `src/model/unet.py` / 配置 | 可配置 InstanceNorm eps；自训练配置用 1e-2，原始模型默认仍为 1e-5 |
| `src/train.py` | 可选择不给归一化 affine 及其前置卷积施加 weight decay |
| `src/transform/vision.py` | FourierDropout min-max 分母 clamp_min(1e-6)；当前新数据训练配置关闭该增强 |
| `src/train.py` | 自定义 checkpoint_dir、无 wrapper 前缀权重、训练 checkpoint 原子 rename、--resume 入口 |

FP32 Muon 的重要前提是 optimizer step 在 autocast 外执行；仅把输入 tensor 转 FP32 而继续在 AMP 上下文内 matmul，不能保证计算实际为 FP32。

## 新增数据

- 新增 `ecg_page_rendering.py` 和三个根目录生成脚本。
- 从 PTB-XL WFDB format 16 和 Chapman 12 导联 CSV 直接渲染页面，并同步产生 RGB soft masks。
- 支持三种布局、按布局选择尺寸池、15 种网格样式和导联连接 continuous / gap / pulse。
- 同一个源记录的布局、尺寸变体及透视 twin 保持相同 split。
- 中文报告文本为程序合成；实际记录的 overlay_probability=0，没有使用私有图像叠加。
- 页面模板参数参考本地扫描几何，发布的生成程序不依赖扫描原图或 XML。
- 划分按源记录组，不是 PTB-XL 官方 patient fold；不能声称已保证同患者不同记录完全隔离。

## 推理扩展

当前工作树另有未提交推理修改，也已复制进发布副本：

- `src/model/cropper.py`：垂直边界可参考网格；透视校正前补画布防止内容裁掉；percentile 索引范围保护。
- `src/model/inference_wrapper.py`：以网格范围为锚点、向相邻信号扩展内容边界，保留高 R 波；限制噪声造成的扩展；基于多行校准脉冲特征清除左侧校准标记。
- `src/digitize.py`：增加布局 / 导联数 / 像素间距 / 校准脉冲位置 metadata，旧 CSV header 不匹配时另存旧文件。
- 本地导联布局与推理模板随代码保留；公开版删除本机路径，新增 `inference_newdata.yml` 指向实际存在的第39轮权重。

这部分是启发式扩展，目前没有随仓库提供独立定量评测，不能据此声称提高了论文 benchmark 分数。

## 发布整理时的修改

发布目录与原工作目录分开。算法代码取自当前工作树；只做公开路径适配和文档整理：

- 两个 Slurm 脚本以环境变量 / 提交目录定位 Python、配置和仓库，移除硬编码账号及 conda 路径。
- 配置的输入 / 输出 / checkpoint 改为仓库中的相对示例路径；修正重复 metrics key。
- 三个生成程序默认 source root 改为 `data/sources`；推荐使用显式 --ptb-root / --chapman-root。
- 上游 redaction 辅助脚本的示例输入 / 输出改为 `data/redaction` / `outputs/redaction`，不携带作者本机目录。
- 增加 LFS attributes、发布目录 ignore 和权重 / 数据 / 训练指标清单。
- 自训练权重仅发布指定的 `npj_newdata_weights_39.pt`；自建图像、mask、源波形及逐条 manifest 不上传，只说明来源、生成脚本与汇总统计。
- README 单独编写，说明复现来源与本项目修改，不复制原仓库 README；保留原项目链接、LICENSE 和论文引用。
- 为替换含敏感信息的旧仓库，发布副本使用全新 Git 历史，报告不包含旧项目 commit 引用；原论文仓库的公开审计版本仍保留用于署名和对照。
- 最终脱敏：移除集群任务编号与本机目录，专用推理配置 / 布局文件改为 `3x4` / `6x2` 等通用名称；示例输入改为相对 `data/` 路径。两份 shell 脚本通过环境变量设置仓库、Python 和缓存，不包含用户名、账号或 conda 路径。
- 路径核验时修正旧推理模板对已不存在的布局文件的引用，改为仓库内的 `lead_layouts_reduced_npj.yml`。
- 未复制本地旧 Git 历史、私有推理批处理脚本、含输出的 redaction notebook、sandbox、缓存和上游自动发版 workflow。

## 已知限制

1. **无独立 test split**：生成报告 test=0，当前 TEST 配置复用 val，不构成独立评估。
2. **非有限梯度仍有发生**：40 轮 newdata 日志记录跳过异常梯度步骤；防护避免坏更新，不等同于消除了根因。
3. **恢复训练不保证连续等价**：外层 AdamMuon 尚未覆盖子 Muon / AdamW state_dict 序列化；保存的 optimizer state 不包含完整动量 / Adam 状态。RNG、数据增强随机状态也未保存。--resume 可恢复模型 / epoch / scheduler 等，但需要进一步修复和验证后才可声称完整恢复。本次提供的权重文件仅供推理。
4. **early stopping**：普通 torchrun 不使用 Ray STOPPER。
5. **验证**：随机裁剪、drop_last=True、分布式 sampler 丢尾，指标不覆盖全部图像的所有像素。
6. **T_max**：每个训练 batch 推进，包括跳过更新的 batch；换卡数 / batch 后需重新考虑调度。
7. **精度**：A100 BF16 配置的每卡 batch 20来自80GB卡，40GB卡需试跑；V100 的 FP32 模式也需减少 batch。
8. **实验 provenance**：历史日志没有保存完整配置 dump / 环境锁。当前 YAML 是可复现配方参考，不保证等于历史每一时刻的文件。只有2卡40轮 newdata日志和对应第39轮文件可确认；当前4卡v2脚本不构成4卡历史结果。
9. **Checkpoint 标注**：state_dict 不含 norm_eps、训练 epoch 或源数据指纹；文件名和训练日志的对应关系来自本地目录，报告对此保留说明。
