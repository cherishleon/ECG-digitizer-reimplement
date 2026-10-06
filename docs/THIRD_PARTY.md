# 上游与数据署名

## 上游代码与模型

原项目：[Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer)。

本仓库保留原始 CC BY-SA 4.0 LICENSE，并在 CHANGES.md标注修改。原始两个07072025模型文件来源于本地原项目weights目录，发布清单包含hash；若需追踪到上游具体模型版本，请对照上游文件，不用本项目训练loss描述原始模型。

发布核验时，上述两个权重的 SHA-256 与原项目审计快照 `97a15087d4abcda843da8c58ee74b1d8f47e6f9a` 中对应文件一致。本仓库 README 单独说明复现与修改内容，不复制原项目 README；原方法及原版使用说明可通过上面的原项目链接查看。

论文：[Digitizing Paper ECGs at Scale](https://doi.org/10.1038/s41746-025-02327-1)，Stenhede、Bjørnstad、Ranjbar，npj Digital Medicine。

## PTB-XL

数据主页：[PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)；该版本文件许可为 CC BY 4.0。历史本地源版本未锁定，不表示已确认使用1.0.3。

请保留PTB-XL作者、数据论文与所用版本引用：
Wagner et al., PTB-XL, a large publicly available electrocardiography dataset,
Scientific Data (2020), DOI:10.1038/s41597-020-0495-6。

## Chapman

历史输入为本地Chapman 12导联CSV集合。对应公开项目参考：
[Chapman–Shaoxing ECG dataset paper](https://doi.org/10.1038/s41597-020-0403-3)。
具体下载版本、转换程序及license尚未由本地报告锁定。这里提供读取程序和生成统计，不分发原始CSV或波形；后续单独分发衍生PNG前应依据实际源版本保留授权和引用。

## 页面与示例

test/test_data、assets、src/transform/overlay_images均来自原项目，用途沿用上游。未复制本地临床输入图片、JPG/XML、私人overlay或notebook输出。生成报告显示private overlay probability=0；合成中文检查信息由程序生成。

代码LICENSE不覆盖或替代第三方源数据的许可。
