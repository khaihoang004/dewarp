# Deshadow Model — Document Shadow Removal Training

Training notebook for a document shadow removal model using the `LoopRepDocEnhanceNet` architecture, designed to run on Kaggle.

## Setup

The notebook performs the following steps automatically when run:

```bash
git clone https://github.com/khaihoang004/dewarp.git
cd dewarp
git fetch origin
git reset --hard origin/main
git clean -fd
pip install -q -r requirements.txt
```

Main libraries used:

- `torch`, `torchvision`
- `albumentations` (image augmentation)
- `kornia` (additional augmentation)
- `pytorch_msssim` (SSIM loss)
- `bitsandbytes` (8-bit AdamW optimizer)
- `wandb` (experiment tracking)
- `opencv-python`, `Pillow`, `matplotlib`, `tqdm`

## Data Structure

Each dataset is expected to follow this directory layout:

```
<dataset_dir>/
├── train/
│   ├── input/      # shadowed images
│   └── target/     # ground-truth shadow-free images
└── test/
    ├── input/
    └── target/
```

Datasets supported in the notebook:

| Dataset  | Class                  | Role |
|----------|-------------------------|------|
| SD7K     | `SD7KTrainDataset` / `SD7KValDataset` | Train + validation set |

## Model

```python
from src.model.deshadow.model import LoopRepDocEnhanceNet
model = LoopRepDocEnhanceNet().to(device)
```


## Loss

```python
from src.loss.deshadow.loss import DocDeshadowLossStage1

criterion = DocDeshadowLossStage1(
    ssim_weight=cfg.loss_ssim_weight,
    loop_weight=cfg.loss_loop_weight,
    kl_weight=cfg.loss_kl_weight,
)
```

Loss is composed of:

- **Charbonnier loss** — main (final) reconstruction loss.
- **SSIM loss** — structural similarity preservation.
- **Loop reconstruction loss** — supervises intermediate loop steps.
- **KL loss** — regularizes the exit/step distribution (entropy/KL) across loop iterations.

## Training

```python
from src.train import train_loop

train_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    cfg=cfg,
)
```
