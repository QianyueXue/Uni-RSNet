# Uni-RSNet Public Code

This folder is a cleaned public release for the final paper version. It keeps
the core Uni-RSNet model, CHR-Bin sampler, SCCO loss, GCC loss, one training
entry, and one final inference entry. Experimental inference scripts and data
conversion scripts are intentionally removed.

## Paper-Aligned Defaults

- Classes: 7 (`lfm`, `sfm`, `bpsk`, `fsk`, `costas`, `qpfm`, `nlfm`)
- Input size: 1024 x 1024
- Query number / Top-K: 100
- Batch size: 8
- Epochs: 100
- Optimizer: AdamW, learning rate 1e-4, backbone learning rate 1e-5
- LR schedule: MultiStepLR, milestones `[30, 60, 90]`, gamma `0.5`
- Gradient clipping: `0.1`
- SCCO temperature: `0.07`
- Loss weights: `loss_vfl=2`, `loss_bbox=5`, `loss_giou=2`, `loss_emb=0.25`, `loss_band=0.25`
- Final inference: confidence `0.80`, same-label IoU merge `0.55`, boundary margin `3`
- CSV preprocessing: `fs=100e6`, `cut_time=0.0002`, `height_fig=1024`,
  `width_fig=2048`, `preferred_window=20`, LAM `lambda=10`, `epsilon=0.01`
- Online clustering: embeddings enabled, similarity threshold `0.99`, EMA momentum `0.95`,
  feature weights `(w_yc, w_h, w_xw, w_cls) = (10, 10, 10, 1)`

## Install

```bash
pip install -r requirements.txt
```

The paper environment is Python 3.8, PyTorch 2.0.0, and CUDA 11.8.

## Inference

Put the final checkpoint at `weights/unirsnet_final.pth`, or pass `--resume`.

```bash
python infer.py --input path/to/signal.csv --resume weights/unirsnet_final.pth
```

The CSV input can be `[time, I, Q]`, `[I, Q]`, or a folder of CSV files:

```bash
python infer.py --input path/to/csv_folder --resume weights/unirsnet_final.pth
```

Outputs are written to `outputs/infer/`:

- `predictions.csv`
- `predictions.json`
- `vis/*_pred.png`

## Training

Use COCO-format data:

```text
data/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

Then run:

```bash
python tools/train.py -c configs/unirsnet.yml
```

To evaluate a checkpoint:

```bash
python tools/train.py -c configs/unirsnet.yml -r weights/unirsnet_final.pth --test-only
```
