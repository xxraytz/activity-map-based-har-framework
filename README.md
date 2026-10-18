# Human Action Recognition Framework

A configuration-driven framework for training, tuning and evaluating neural networks on skeletal activity maps. It supports different architectures, data representations, label mappings and evaluation setups through YAML configurations, with Optuna hyperparameter optimization.

[Paper](https://doi.org/10.18287/2412-6179-CO-1522) · [AMG — dataset preparation](https://github.com/xxraytz/amg)

## Features

- **Model selection:** CNN and transformer backbones through timm, plus custom ViT, MLP, late-fusion CNN and skeleton 1D CNN models.
- **Experiment configuration:** input channels and dimensions, temporal window size and stride, label mappings, normalization, batch size, optimizer settings and learning-rate scheduling.
- **Hyperparameter optimization:** Optuna search over learning rate, weight decay, optional window stride and candidate architectures; repeated trials, pruning, persistent studies and export of trial results and best parameters.
- **Training:** checkpointing, resume, early stopping and configurable metrics; multiclass soft targets and multilabel training.
- **Evaluation:** leave-one-person-out or explicit manifest splits; window, frame and annotated-segment metrics; multiple overlap-pooling and smoothing methods. A separate TST evaluator produces whole-video decisions.

## Workflow

Prepare activity maps with **AMG**, describe the dataset and experiment in YAML, optionally run HPO, then train and evaluate. Model architecture and experiment settings are selected in the configuration.

```bash
pip install -r requirements.txt

# Train an experiment described by a YAML configuration.
python -m har train --config configs/config.yaml --run-dir runs/my-experiment --device cuda:0

# Tune a multiclass experiment on its validation split.
python -m har tune \
  --config configs/config.yaml \
  --storage sqlite:///optuna.db --study-name har-tuning \
  --trials 40 --max-epochs 30 --repeats 3 \
  --optimize val_loss --direction minimize --device cuda:0

# Evaluate temporal pooling and smoothing for a trained run.
python -m har evaluate temporal --run-dir runs/my-experiment --device cuda:0
```

HPO also accepts `--model-choices` (comma-separated model names) and `--include-window-stride`. Input dimensions must be compatible with each candidate architecture. The supplied configs are experiment examples; set their dataset paths, classes and splits for your data.

## Experiment: TST Fall Detection v2

The results below use **ResNet18 trained from scratch, five grayscale channels, 32 × 32 windows, and stride 1**.

**TST Fall Detection v2:** 264 recordings, 11 people, eight activities. Evaluation uses leave-one-person-out cross-validation with seeds 42, 43 and 44. Each fold contains 216 training, 24 validation and 24 test videos. Normalization uses training data; checkpoints are selected by validation loss.

| Method / work | Input | Evaluation unit / task | Classes | Test protocol | Accuracy, % |
| --- | --- | --- | ---: | --- | ---: |
| [Gasparrini et al.](https://doi.org/10.1007/978-3-319-25733-4_11), first algorithm | Skeleton data | Binary fall detection across eight scenarios | 2 | Split not specified | 79 |
| [Manzi et al.](https://doi.org/10.3390/s17051100), Table 6 | Skeleton data | Overlapping 100-frame clips; ADL and falls evaluated separately | 4 + 4 | New-person evaluation | 100 (ADL); 93.3 (falls) |
| [Yin et al., MC-LSTM](https://doi.org/10.1109/TBCAS.2021.3064841) | Skeleton data | Exact unit not verified in the primary source | 8 | LOPO | 93.9 |
| [Our published method (2025)](https://doi.org/10.18287/2412-6179-CO-1522), earlier taxonomy | Activity maps | Frame | 9 | LOPO | 83.2 |
| **This experiment: raw network output** | Activity maps | Window | 9 | LOPO, 3 seeds | **85.52 ± 0.63** |
| **This experiment: max pooling + moving average** | Activity maps | Frame | 9 | LOPO, 3 seeds | **87.57 ± 0.75** |
| **This experiment: max pooling + hysteresis + vote** | Activity maps | Whole video | 8 | LOPO, 3 seeds | **93.31 ± 1.16** |

Mean ± sample standard deviation across three seeds.

## Data format and TST preparation

For the TST experiment, the video classes are `adl_sit`, `adl_grasp`, `adl_walk`, `adl_lay`, `fall_front`, `fall_back`, `fall_side`, and `fall_endupsit`. Frame/window labels additionally include `no_class`.

The preparation script maps existing frame annotations to the source recording's activity. For example, `falling` and `sitting` become `fall_endupsit` only inside recordings of that scenario. Other frames become background. The full mapping is defined in `har/data/preparation/tst.py`; source identity is used for target construction, never as a model input.

Prepare activity maps using [AMG](https://github.com/xxraytz/amg). Required inputs:

- TST source skeleton recordings and the original frame-annotation JSON;
- NPZ sequences containing `data` shaped `(C, H, T)` (the reported TST experiment uses `C=5`, `H=32`) and `labels` of length `T`;
- a Parquet manifest with `filepath`, `actor_id`, and `seq_id`.

Paths in the manifest resolve from the repository root. Data, original annotations and model weights are not included. Reproducing the reported scores requires the matching AMG maps and annotations.

## Experiment configuration

Each training run is defined by a YAML file. Start from a configuration in [`configs/`](configs/) and adapt it to your dataset and experiment. The same configuration is used by `har train` and as the starting point for `har tune`.

| Area | Configuration fields | Purpose |
| --- | --- | --- |
| Architecture | `model.name`, `model.pretrained`, model-specific options | Select a timm backbone or a custom model |
| Input | `in_chans`, `img_size`, `data.window_size`, `data.window_stride` | Set the activity-map shape and temporal sampling |
| Dataset and labels | `data.manifest`, `data.class_names`, `num_classes` | Locate sequences and map raw labels to output classes |
| Validation protocol | `data.split_mode`, `data.lopo`, `data.val_frac` | Hold out a person or use explicit manifest splits |
| Normalization | `data.normalize` | Select preprocessing, including training-set channel statistics |
| Training | `batch_size`, `epochs`, `lr`, `weight_decay`, `training` | Set optimization, scheduling, early stopping and checkpoint selection |
| Metrics | `metrics.names`, `metrics.eval_every` | Choose metrics and evaluation frequency |
| Runtime | `seed`, `device`, `num_workers` | Control initialization and execution |
| HPO candidates | `optuna.model_names` | Optionally include architecture selection in an Optuna study |

For example, these fields select a backbone and a subject-independent split **within a complete training configuration**:

```yaml
model:
  name: resnet50
  pretrained: false

in_chans: 5
img_size: 32

data:
  manifest: data/my-dataset/manifest.parquet
  window_size: 32
  window_stride: 1
  split_mode: lopo
  lopo: 1
  val_frac: 0.1
  normalize:
    mode: trainset_channel_standard

optuna:
  model_names: [resnet18, resnet50]
```

With `split_mode: lopo`, `data.lopo` identifies the test person; validation is drawn from the remaining people. With `split_mode: manifest`, the manifest specifies train, validation and test membership.

`data.class_names` maps raw annotation labels to output classes. `num_classes` must match the number of distinct output classes. Input dimensions and model-specific options must be compatible with the selected architecture.

HPO searches learning rate and weight decay, with optional architecture and window-stride search. Trial count, objective, repetitions and study storage are CLI options; use `python -m har tune --help` to see them. The search exports trial results and best parameters.

Temporal evaluation has a separate [configuration](configs/evaluate/temporal_pooling.yaml) for overlap pooling, smoothing and segment decisions. The [TST training template](configs/tst-video8-template.yaml) records the configuration used for the results above.

Use a separate run directory for each experiment. The run saves its resolved configuration alongside checkpoints and metrics; `--resume` continues an existing run.

## Package structure

```text
har/
  data/            Sequence loading, labels, normalization and dataset preparation
  models/          Model factory and custom architectures
  training/        Training loop, checkpoints and resume
  tuning/          Optuna optimization
  evaluation/      Window, frame, segment and video evaluation
  visualization/   Confusion matrices and prediction plots
  experiments/     TST experiment orchestration
  cli.py           Command dispatch
configs/           Experiment configurations
tests/             Pipeline and compatibility checks
```

Use `python -m har --help` to list commands, and e.g. `python -m har plot --help` for visualization tools. An editable installation (`pip install -e .`) also provides the `har` command. Run experiments from the repository root; `HAR_PROJECT_ROOT` can select another workspace for dataset-specific commands.

Legacy `main.py`, `src` imports and `scripts/*.py` entry points remain as thin compatibility wrappers. Their implementations live in `har`; existing launch commands and state-dict checkpoints remain supported. Local orchestration under `scripts/local/` is retained separately.

## Citation

E. E. Surkov, O. S. Seredin, A. V. Kopylov. **Human Action Recognition Based on the Skeletal Pairwise Dissimilarity.** Computer Optics, 49(3), 493–503, 2025. [DOI](https://doi.org/10.18287/2412-6179-CO-1522).

## License

[MIT](LICENSE). Dataset usage is subject to its own terms.
