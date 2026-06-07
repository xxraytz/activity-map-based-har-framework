# human-action-recognition
Skeleton based approach to Human action recognition with CNN, ViT, SwinT etc

## Temporal Evaluation Protocol

### Goal

Evaluate trained window-level HAR runs at three levels:

1. raw/window predictions;
2. frame-level predictions reconstructed from overlapping windows;
3. episode-level activity decisions over continuous ground-truth segments.

### Run Command

Use the default evaluation config:

```bash
python3 scripts/evaluate_temporal_pooling.py --run-dir runs/<run_name>
```

Optional overrides:

```bash
python3 scripts/evaluate_temporal_pooling.py \
  --run-dir runs/<run_name> \
  --eval-config configs/evaluate/temporal_pooling.yaml \
  --device cuda:0 \
  --split test
```

The output directory is:

```text
runs/<run_name>/temporal_eval/
```

### Current Config

The current config computes:

```text
pooling.methods = [max]
smoothing.methods = [moving_average, hysteresis]
segments.decision_methods = [mean_prob]
```

The evaluator combines these lists as:

```text
for pooling in pooling.methods:
  reconstruct frame probabilities from overlapping windows
  for smoothing in smoothing.methods:
    compute frame metrics
    for decision in segments.decision_methods:
      compute episode metrics
```

Therefore the current evaluation produces:

```text
frame:
  max + moving_average
  max + hysteresis

episode:
  max + moving_average + mean_prob
  max + hysteresis + mean_prob
```

### Metric Sources

Raw/window metrics come from the normal training pipeline:

```text
runs/<run_name>/metrics.csv
```

Temporal frame metrics come from:

```text
runs/<run_name>/temporal_eval/temporal_metrics.csv
```

Episode metrics come from:

```text
runs/<run_name>/temporal_eval/segment_metrics.csv
```

### Summary Columns

Use these columns for final experiment summaries:

| column | source |
|---|---|
| `window/raw f1` | `metrics.csv`, `split=test`, `metric=f1_macro` |
| `frame pooled f1` | `temporal_metrics.csv`, `pooling=max`, `smoothing=moving_average` |
| `episode soft f1` | `segment_metrics.csv`, `pooling=max`, `smoothing=moving_average`, `decision=mean_prob` |
| `episode hysteresis f1` | `segment_metrics.csv`, `pooling=max`, `smoothing=hysteresis`, `decision=mean_prob` |

### Interpretation

`max + moving_average + mean_prob` is the main soft episode metric. It keeps probabilities soft, smooths them over frames, then averages probabilities inside each true activity segment.

`max + hysteresis + mean_prob` is the CUSUM/PAGE-like diagnostic metric. It applies stateful class switching after frame reconstruction, then averages the resulting segment probabilities.
