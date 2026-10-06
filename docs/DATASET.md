# 新增合成 ECG 数据

推荐入口是 `generate_open_ecg_gridtype_dataset.py`。该文件导入另外两个生成文件以及 `ecg_page_rendering.py`，四个文件必须保留在一起。

| 脚本 | 作用 |
| --- | --- |
| `generate_open_ecg_gridtype_dataset.py` | 当前数据生成入口：三种布局、15 种网格样式、增强、train/val/test 划分 |
| `generate_open_ecg_segmentation_dataset.py` | 三种页面布局与 RGB soft mask 渲染逻辑 |
| `ecg_page_rendering.py` | 页面几何、网格及同步标签合成 |
| `generate_synthetic_3x4_rhythm_dataset.py` | PTB-XL / Chapman 波形读取及旧版渲染工具；旧六通道 mask 不直接用于当前训练 |

## 波形来源和格式

- PTB-XL：递归发现 .hea，默认仅 high resolution，读取配套 format-16 .dat，使用 header 的 gain/baseline 转为 mV。
- Chapman：读取 source root 顶层 CSV，支持 Nx12 或 12x5000，采样率按500Hz解释。导联顺序为 I, II, III, aVR, aVL, aVF, V1–V6。默认输入 µV，除1000转mV；需确认自己的源 CSV 单位。
- 缺失值会按有效样本中位数填充；全缺失导联拒绝生成。曲线去中位数、按设定 mV 范围截断。
- 源版本和下载校验和在历史生成报告中没有记录。本仓库不把“任意最新版源数据”宣称为与历史生成完全一致。

历史数据共32387源记录、97161渲染job，每条源记录3种布局；额外透视 twin 增加19339页，总116500页。详见 `reports/generation_report.json`。

## 布局及尺寸（宽×高）

| 标识 | 内容 | 尺寸池 |
| --- | --- | --- |
| 4x3_plus_3 | 3行×4列短导联 + V1 / II / V5 长节律 | 3508×2480、3320×2552 |
| 6x2_plus_1 | 6行×2列短导联 + II 长节律 | 3508×2480、1667×1261 |
| 4x3_plus_1 | 3行×4列短导联 + II 长节律 | 3508×2480、1667×1261 |

网格样式15种，含彩色实线、灰白、点阵、虚线、仅粗格和无网格。默认导联过渡 continuous/gap/pulse 权重0.2/0.4/0.4；10秒页面，25mm/s，10mm/mV，幅度比例0.85–1.15。

## 文件和标签

```text
data/npj_ecg_dataset_new/
  generation_report.json
  manifest.csv
  train/<stem>.png
  train/<stem>_mask.png
  val/<stem>.png
  val/<stem>_mask.png
  test/                     # 历史版本为空
```

Mask 是 RGB PNG soft target：红=网格，绿=文字/校准/条码等干扰，蓝=生理曲线。不是整数类别索引，也不是旧生成脚本的六通道稀疏 NPZ。训练中的 rgb_to_one_hot 对通道总和超过1的像素归一化，背景为1减归一化后的通道和。

生成时会在自己的输出目录创建 `manifest.csv`，记录图像、mask、布局和增强参数。这个逐条清单不随 GitHub 仓库发布，汇总统计见 `reports/generation_report.json`。重新生成需自行准备源波形，仓库不包含源数据。

## 默认配方

seed=1234；split_ratios=0.85,0.15,0；variants_per_record=1；中文文本概率0.8；条码0.25；空白局部遮挡0.1；摩尔纹0.2；眩光0.15；extra_perspective_probability=0.2；base perspective_probability=0；distortion=0.03；rotation=4°；private overlay=0；blank_page_ratio=0。

split通过seed和源记录组的稳定hash决定。PTB高/低采样率以及同一源的 blank/layout/perspective变体按同组处理。**这是记录级划分，不是患者级划分**；PTB-XL同患者其他记录可能落入另一split。独立泛化实验建议利用官方患者fold另行划分，并为新实验保存配置及 manifest。

历史没有test，也没有独立测试指标。若使用0.8/0.1/0.1，新数据版本及分割会改变，应另存输出目录并训练新模型。

## 分发范围

这里仅分发生成代码、数据来源说明、生成参数及汇总统计，不发布116500对PNG、逐条manifest、原始临床扫描或源波形。中文姓名/检查号由程序合成，不是从真实报告拷贝。本地几何模板只包含布局参数；私有扫描图不包含在发布副本中。源数据的许可证和引用必须保留，见 THIRD_PARTY.md。
