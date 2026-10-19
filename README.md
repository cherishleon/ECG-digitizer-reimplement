# ECG-digitizer Reproduction

**English** | [简体中文](README.zh-CN.md)

This repository reproduces the ECG image digitization method described in *Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research*, using the authors' [Open-ECG-Digitizer code](https://github.com/Ahus-AIM/Open-ECG-Digitizer) for training and inference.

The reproduction retains the original segmentation network and digitization pipeline, while adding multi-GPU training, V100 / A100 support, numerical safeguards during training, and inference improvements. It also includes scripts for generating new synthetic training data from PTB-XL / Chapman waveforms and provides segmentation weights trained on that data. The sections below describe the modifications, data generation process, and usage.

## Changes to the Original Project

- **Multi-GPU training**: Added DDP, DistributedSampler, automatic multi-GPU launch, and explicit `torchrun` support in `src/train.py` / `src/utils.py`. Only rank 0 saves models.
- **GPU and numerical support**: Muon uses FP32 on V100 and supports BF16 on A100. Loss is computed in FP32, with gradient clipping, non-finite value safeguards, and configurable InstanceNorm eps.
- **Custom data**: Added page rendering and data generation scripts that produce ECG images and RGB masks in three layouts and 15 grid styles from PTB-XL / Chapman waveforms. Only the scripts and documentation are included; the generated images are not uploaded.
- **Inference improvements**: Adjusted cropping and signal content boundaries, added calibration pulse handling, and expanded output metadata.
- **Training and inference configurations**: Provided single-GPU / multi-GPU configurations, two A100 Slurm scripts, configurable checkpoint directories, and inference configurations matching the new weights.
- **Weights trained in this reproduction**: Released `npj_newdata_weights_39.pt`, with clear attribution distinguishing it from the original project's models.

See [docs/CHANGES.md](docs/CHANGES.md) for changes by file and [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) for attribution of the original method and models.

## Installation

Python 3.12 is required. GPU training requires a PyTorch CUDA build compatible with your driver. The environment available when this repository was prepared used Python 3.12.0 and PyTorch 2.6.0+cu124; this does not mean that a complete dependency lock was saved for the historical training jobs.

```bash
git clone https://github.com/<YOUR_USERNAME>/ECG-digitizer.git
cd ECG-digitizer
git lfs install
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` retains the original project's dependencies. `reports/environment_snapshot.json` is a snapshot taken when this repository was prepared, not a guarantee of an identical training environment. Select a PyTorch build appropriate for your CUDA environment.

## New Synthetic Data

This reproduction uses **public 12-lead WFDB waveforms from PTB-XL** and **12-lead CSV waveforms from Chapman** to render ECG pages with grids, lead labels, and augmentations. Matching segmentation masks are generated for training the segmentation model.

The main entry point is `generate_open_ecg_gridtype_dataset.py`, supported by `generate_open_ecg_segmentation_dataset.py` (layouts and masks), `ecg_page_rendering.py` (pages and grids), and `generate_synthetic_3x4_rhythm_dataset.py` (waveform loading utilities). These scripts are included in the repository and can be run with your own source data paths.

This reproduction generated **116,500 PNG image / mask pairs** from 32,387 source records:

| Item | Count |
| --- | ---: |
| train | 98,694 |
| val | 17,806 |
| test | 0 |
| PTB-XL-derived pages | 78,396 |
| Chapman-derived pages | 38,104 |
| Frontal pages | 97,161 |
| Additional perspective pages | 19,339 |

The layouts are `4x3_plus_3`, `6x2_plus_1`, and `4x3_plus_1`. Here, `4x3` means 4 columns × 3 rows, commonly called a 3×4 ECG page. Mask channels are R=grid, G=text / calibration pulses / interference marks, and B=physiological ECG traces; the background is derived from the remaining probability.

**The generated images, masks, raw waveforms, and per-record manifests are not uploaded to GitHub.** Only the generation code, data sources, recipe, and summary statistics are provided. See the [dataset documentation](docs/DATASET.md) for parameters, waveform formats, splitting, and limitations. The upstream project's fixed augmentation assets and small test samples are retained so that the code and tests can run; they are separate from the data generated for this reproduction.

The historical generation used a record-level 85% / 15% / 0% split. The TEST configuration currently reuses val to satisfy the existing loader interface; **there are no independent test-set results**. To create an independent test set, regenerate the data with a split such as `--split-ratios 0.8,0.1,0.1` and update the TEST path. This changes the data split and training results.

After downloading the source waveforms, start with a small generation run:

```bash
python generate_open_ecg_gridtype_dataset.py --ptb-root /path/to/ptb-xl --chapman-root /path/to/chapman_csv --output-dir data/smoke_dataset --limit-ptb 2 --limit-chapman 2 --workers 2 --seed 1234
```

To reproduce the historical generation recipe:

```bash
python generate_open_ecg_gridtype_dataset.py --ptb-root /path/to/ptb-xl --chapman-root /path/to/chapman_csv --output-dir data/npj_ecg_dataset_new --workers 24 --seed 1234 --split-ratios 0.85,0.15,0
```

Chinese report text requires a Chinese font, supplied with `--chinese-font /path/to/font.ttf`. Without a font, use `--chinese-text-probability 0` for a functional check, but this no longer matches the historical recipe. Chapman input must be 500 Hz, 12-lead CSV data; the default unit conversion divides values by 1000. Do not pass WFDB data without first confirming its format.

## Single-GPU / Multi-GPU Training

Run these commands from the repository root. Prefer the A100 BF16 configurations. Reduce the batch size as needed for GPUs with 40 GB of memory; the historical jobs used A100-SXM4-80GB GPUs.

```bash
# Expose one GPU for single-process training
CUDA_VISIBLE_DEVICES=0 python -m src.train --config src/config/unet_a100_newdata.yml

# Match the historical newdata script: 2 A100 GPUs
torchrun --standalone --nproc-per-node=2 --module src.train --config src/config/unet_a100_newdata.yml

# Current v2 script: 4 A100 GPUs
torchrun --standalone --nproc-per-node=4 --module src.train --config src/config/unet_a100_v2.yml

# Automatically use all visible GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train --config src/config/unet_a100_v2.yml
```

`DATALOADER.KWARGS.batch_size` is the **batch size per GPU**. Global batch size = batch size per GPU × number of GPUs. Gradient accumulation is not currently implemented.

| Configuration / script | GPUs | Batch per GPU | Global batch | Epochs | AMP | Initial LR | T_max |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| newdata / `run_a100_newdata.sh` | 2 | 20 | 40 | 40 | BF16 | 0.0037 | 46,500 |
| v2 / `run_a100_v2.sh` | 4 | 20 | 80 | 40 | BF16 | 0.0037 | 46,500 |

T_max is counted in training batches. The minimum LR is 0.00037 and remains constant after it is reached. Batches with skipped optimizer updates still advance the scheduler. For 98,694 images, 2 GPUs yield approximately 2,467 steps per epoch and reach the minimum LR around epoch 18.85; 4 GPUs yield approximately 1,233 steps per epoch and reach it around epoch 37.71. Changing the GPU count therefore also changes how many epochs the cosine schedule spans. These two configurations are not fully equivalent experiments.

The new data paths use forms such as `../../data/npj_ecg_dataset_new/train` because the existing ECGScanDataset resolves relative paths from `src/dataset/`. You can also replace them with your own absolute paths.

Set `TRAIN.KWARGS.checkpoint_dir` to choose the output directory, relative to the repository root:

```yaml
TRAIN:
  KWARGS:
    checkpoint_dir: 'outputs/a100_newdata'
```

Only rank 0 saves `weights_<epoch>.pt`, without DDP / compile prefixes. Standard training does not use the Ray `STOPPER` in the YAML and runs for the configured number of epochs. Validation uses random crops and defaults to drop_last=True, so it is not deterministic, full-image validation.

The V100 example is `src/config/unet.yml`, which uses FP32 forward passes and FP32 Muon. Its data paths point to an older data directory that you must prepare yourself. Do not assume that a batch size of 8 per GPU will fit in 32 GB of memory with FP32; start with a small batch for a trial run. V100 and A100 GPUs cannot be mixed in one job because the current Muon implementation selects different computation dtypes for these GPUs.

Use `--resume` to resume training from `checkpoint_last.pt` in the output directory. The provided epoch 39 file contains only the model state_dict and can be used for inference, but cannot restore the full training state. The sub-optimizer states of the custom AdamMuon are not yet fully included in the outer optimizer's serialization, and random number generator states are not saved. Resumed training is therefore not guaranteed to match uninterrupted training. See the [known limitations](docs/CHANGES.md).

## Slurm

Activate your environment before submitting jobs on the cluster. The scripts do not hardcode user directories or project accounts:

```bash
mkdir -p logs
sbatch --account=<YOUR_ACCOUNT> --export=ALL,ECG_PYTHON=/path/to/env/bin/python run_a100_newdata.sh
sbatch --account=<YOUR_ACCOUNT> --export=ALL,ECG_PYTHON=/path/to/env/bin/python run_a100_v2.sh
```

The repository path is taken from `ECG_REPO_DIR` first, then `SLURM_SUBMIT_DIR`. Outside Slurm, it defaults to the script's directory. Set `ECG_PYTHON`, `ECG_CONFIG`, and `ECG_TRITON_CACHE_DIR` to override the Python executable, configuration, and cache directory, respectively. Adjust the GPU type name `a100`, resource requests, and log directory for your cluster. Supply the account at submission time rather than writing it into the scripts.

## Model Weights and Inference

The only segmentation weights trained in this reproduction that are released are `npj_newdata_weights_39.pt`, trained on the new synthetic data. Weights from other training epochs are not released. The original project's models are also retained and clearly attributed.

| File | Purpose | Source |
| --- | --- | --- |
| `checkpoints/npj_newdata_weights_39.pt` | Four-class ECG page segmentation trained in this reproduction | newdata, epoch 39, validation loss 0.0435144846 |
| `checkpoints/unet_weights_07072025.pt` | Original four-class page segmentation | Original upstream file |
| `checkpoints/lead_name_unet_weights_07072025.pt` | Lead label recognition | Original upstream file; not retrained |

See the [checkpoint documentation](checkpoints/README.md) for details and `reports/checkpoints.json` for verification metadata. If Git LFS downloads fail, the files remain text pointers and cannot be loaded directly with torch.load.

```bash
python -m src.digitize --config src/config/inference_newdata.yml DATA.images_path=/path/to/ecg_images DATA.output_path=outputs/inference
```

The weights trained in this reproduction require `norm_eps: 1.0e-2`, which is not stored in the state_dict. The original segmentation weights use `1.0e-5`. The published `inference_newdata.yml` already sets the correct value. Use `lead_layouts_*.yml` to restrict the layouts considered during recognition. Output waveforms are in μV.

For fixed layouts, the repository also provides `inference_3x4_newdata.yml` (3×4 short leads + 3 long rhythm strips) and `inference_6x2_newdata.yml` (6×2 short leads + 1 long rhythm strip). These configuration names describe the layouts and do not include local data source labels.

The complete local logs record training on 2×A100-SXM4-80GB for 40 epochs. Epoch 39 had train_loss=0.0573955938 and val_loss=0.0435144846; epoch 40 had val_loss=0.0435892576. These are validation losses on random crops of synthetic data, not independent test results or waveform reconstruction accuracy. The logs also record several steps skipped due to non-finite gradients, so they do not establish that every training step was stable. Per-epoch metrics are available in `reports/training_newdata.csv`.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `src/train.py` | DDP launch, precision control, training / validation, checkpointing |
| `src/utils.py` | DistributedSampler, cosine-to-constant LR schedule, rank 0 plotting |
| `src/model/unet.py` | Segmentation network, configurable norm_eps, freezing unused layers |
| `src/optimizer/adammuon.py` | Muon + AdamW, dtype selection, parameter-group weight decay |
| `src/transform/vision.py` | Data augmentation, division-by-zero protection in Fourier normalization |
| `ecg_page_rendering.py` | Page geometry, grid and mask composition |
| `generate_open_ecg_gridtype_dataset.py` | Recommended data generation entry point |
| `generate_open_ecg_segmentation_dataset.py` | Three-layout rendering logic used by gridtype generation |
| `generate_synthetic_3x4_rhythm_dataset.py` | Waveform loading and legacy rendering utilities; the old six-channel masks cannot be used directly for current training |
| `src/model/cropper.py`, `inference_wrapper.py` | Content preservation, cropping improvements, calibration pulse handling |
| `docs/`, `reports/` | Changes, dataset and release documentation, and verifiable experiment records |

## Local Checks

```bash
python -m pip install pytest
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q test
python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/smoke_ddp.py
```

The tests generate temporary weights in the ignored `sandbox/` directory. CPU/Gloo checks do not replace actual GPU training or end-to-end accuracy evaluation. The results recorded when this repository was prepared are in `reports/release_validation.json`.

## Original Work, License, and Citation

The original method and implementation come from [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer). This repository reproduces and modifies that work, retaining the original project's [CC BY-SA 4.0 LICENSE](LICENSE). New code is provided under the same license. Data sources retain their respective licenses and citation requirements; see the [third-party documentation](docs/THIRD_PARTY.md). Original clinical JPG/XML files, private overlay images, patient records, cluster credentials, and training logs are not distributed with this repository.

Please cite the original paper:

```bibtex
@article{stenhede_digitizing_2026,
  title = {Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research},
  author = {Stenhede, Elias and Bjørnstad, Agnar Martin and Ranjbar, Arian},
  journal = {npj Digital Medicine},
  year = {2026},
  doi = {10.1038/s41746-025-02327-1}
}
```

See [docs/DATASET.md](docs/DATASET.md) for detailed data generation documentation and [docs/CHANGES.md](docs/CHANGES.md) for changes by file and current limitations.
