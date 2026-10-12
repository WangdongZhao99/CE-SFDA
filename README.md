# CE-SFDA

**Classifier Ensemble based Source-free Domain Adaptation for Time Series Classification**

This repository provides the implementation of **CE-SFDA**, a classifier-ensemble method for source-free domain adaptation (SFDA) in time-series classification. The model is first trained with labeled source-domain data and then adapted using unlabeled target-domain data, without accessing the source data during adaptation.

The repository also includes several established SFDA baselines, configurable time-series datasets, training hyperparameters, evaluation code, and checkpoint export.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/WangdongZhao99/CE-SFDA.git
cd CE-SFDA

# Install a PyTorch/torchvision version compatible with your CUDA driver.
pip install torch torchvision
pip install -r requirements.txt
pip install torchmetrics wandb skorch
```

Python 3.8–3.10 is recommended. For a CPU-only run, replace `--device cuda` with `--device cpu`.

## Datasets

The data loader expects one PyTorch file for each domain and split. Set `--data_path` to the dataset root; the code automatically appends the selected dataset name.

```text
<data_path>/
└── <DATASET>/
    ├── train_<domain_id>.pt
    └── test_<domain_id>.pt
```

Each file should be saved with `torch.save` and contain at least:

```python
{
    "samples": samples,  # torch.Tensor or numpy.ndarray
    "labels": labels      # torch.Tensor or numpy.ndarray
}
```

The recommended sample shape is `(N, C, L)`, where `N` is the number of samples, `C` is the number of channels, and `L` is the sequence length. The loader also accepts two-dimensional samples and three-dimensional samples whose channel dimension is last.

The following datasets and domain scenarios are configured in `trainers/configs/data_model_configs.py`:

| Dataset | Classes | Channels | Sequence length | Default scenarios |
| --- | ---: | ---: | ---: | --- |
| `WISDM` | 6 | 3 | 128 | `20→30`, `35→31`, `6→19`, `2→11`, `23→32` |
| `HHAR` | 6 | 3 | 128 | `1→6`, `3→8`, `4→5`, `8→3`, `7→4` |
| `FD` | 3 | 1 | 5120 | `0→1`, `1→2`, `3→1`, `1→0`, `2→3` |
| `HAR` | 6 | 9 | 128 | `2→11`, `6→23`, `7→13`, `9→18`, `12→16` |
| `EEG` | 5 | 1 | 3000 | `0→11`, `12→5`, `7→18`, `16→1`, `9→14` |

For example, the default HAR scenario `2→11` requires:

```text
D:/dataset/HAR/train_2.pt
D:/dataset/HAR/test_2.pt
D:/dataset/HAR/train_11.pt
D:/dataset/HAR/test_11.pt
```

## Quick start

Run the following command from the repository root:

```bash
python trainers/train.py \
    --data_path D:/dataset \
    --dataset HAR \
    --da_method CESFDA \
    --backbone CNN \
    --device cuda \
    --num_runs 3 \
    --run_description har_cesfda
```

A minimal CPU example is:

```bash
python trainers/train.py \
    --data_path ./data \
    --dataset HAR \
    --da_method CESFDA \
    --backbone CNN \
    --device cpu \
    --num_runs 1
```

For every configured source-to-target scenario, the trainer:

1. pre-trains a classifier on the labeled source-domain training set;
2. adapts the model with the unlabeled target-domain training set;
3. evaluates the adapted model on the target-domain test set; and
4. saves metrics, risks, logs, and model checkpoints.

## Methods and model options

The implemented adaptation methods include:

```text
SHOT, CESFDA, AaD, NRC, MAPU, SCLM, TPDS, GKD
```

`COWA` is configured for some datasets. The available methods depend on the selected dataset because their hyperparameters are defined separately in `trainers/configs/hparams.py`.

The current checkout registers `CNN` in `trainers/models/models.py`. The argument help text also mentions `RESNET18` and `TCN`, but those classes are not present in the current model file; select `CNN` unless you implement and register the additional backbones first.

## Main arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--data_path` | `D:/dataset` | Root directory containing dataset folders |
| `--dataset` | `WISDM` | `WISDM`, `HHAR`, `FD`, `HAR`, or `EEG` |
| `--da_method` | `CESFDA` | Domain-adaptation method |
| `--backbone` | `CNN` | Registered feature extractor |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--num_runs` | `3` | Number of runs with different random seeds |
| `--save_dir` | `experiments_logs_0.3` | Output directory name |
| `-run_description` | selected method | Experiment identifier |
| `--is_weighted` | `False` | Use entropy-based weighted ensemble evaluation |
| `--test_method` | `ensemble` | `fixed`, `learnable`, or `ensemble` |

Edit `trainers/configs/data_model_configs.py` to change domain scenarios or dataset dimensions. Edit `trainers/configs/hparams.py` to change epoch counts, batch sizes, learning rates, and method-specific hyperparameters.

## Outputs

For a run started from the repository root, results are written under the parent directory of the repository:

```text
experiments_logs_0.3/
└── <dataset>/
    └── <run_description>/
        ├── <source>_to_<target>_run_<run_id>/
        │   ├── checkpoint.pt
        │   └── logs_<timestamp>.log
        ├── results.csv
        └── risks.csv
```

`results.csv` contains target-domain accuracy, macro-F1, and AUROC. `risks.csv` contains source and target cross-entropy risks. Each checkpoint stores the non-adapted, last adapted, and best adapted model states.

## Repository structure

```text
trainers/
├── train.py                 # Training and evaluation entry point
├── abstract_trainer.py      # Training loop, evaluation, and checkpointing
├── algorithms/algorithms.py # SFDA algorithms
├── configs/                 # Dataset and hyperparameter definitions
├── dataloader/              # .pt loading and normalization
├── models/                  # Feature extractors and classifiers
└── scripts/                 # Historical shell examples
```

The shell files in `trainers/scripts/` contain examples from earlier experiments and include arguments that are not accepted by the current `train.py`. The direct command shown above is the recommended entry point.

## Citation

If this code is useful in your research, please cite the corresponding CE-SFDA paper and include the official bibliographic information when it becomes available.

