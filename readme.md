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
from src.deshadow.model.model import LoopRepDocEnhanceNet
model = LoopRepDocEnhanceNet().to(device)
```


## Loss

```python
from src.deshadow.loss.loss import DocDeshadowLossStage1

criterion = DocDeshadowLossStage1(
    ssim_weight=cfg.loss_ssim_weight,
    loop_weight=cfg.loss_loop_weight,
    kl_weight=cfg.loss_kl_weight,
)
```

Loss is composed of:

- **Charbonnier loss** — main (final) reconstruction loss.
- **Loop reconstruction loss** — supervises intermediate loop steps.
- **KL loss** — regularizes the exit/step distribution (entropy/KL) across loop iterations.

## Training

```python
from src.deshadow.train.train import train_loop

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
## Running the Demo

### Step 1: Prepare the Directory Structure
Ensure your project folder contains both the model weights and your frontend files in their correct default locations:
```text
dewarp/
├── main.py
├── src/deshadow/test/checkpoint/latest_checkpoint(0107).pth  <-- Ensure weights are here
└── static/
    └── index.html
```

### Step 2: Install Dependencies
```
!pip install -q -r requirements.txt
```

### Step 3: Start the Server
Execute the server script directly from your terminal:
```
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Step 4: Access the Application
Open your web browser and navigate to:
`http://127.0.0.1:8000`









## Running the Demo (Using Kaggle as a GPU Server)

This system is designed to let you leverage the free T4 GPU on Kaggle as the AI processing server, while your Web UI runs on your local machine via an Ngrok tunnel.

### Step 1: Set up Environment Variables (Kaggle Secrets)
Before running the code, you need to configure secure environment variables on Kaggle. 
1. Open your Kaggle Notebook.
2. On the top menu, navigate to **Add-ons** > **Secrets**.
3. Add the following 2 variables and **make sure to check the "Attach" box**:
   - `NGROK_AUTH_TOKEN`: Get your free token from the [Ngrok Dashboard](https://dashboard.ngrok.com/).
   - `CKPT_PATH`: The absolute path to your model weights on Kaggle.

### Step 2: Start the Kaggle Server
Open a new code cell on Kaggle, copy the following commands, and run them. This will automatically clone the source code, install dependencies, and start the FastAPI server alongside Ngrok:

```bash
!test -d dewarp || git clone https://github.com/khaihoang004/dewarp.git
%cd dewarp
!pip install -q -r requirements.txt
!uvicorn server:app --host 0.0.0.0 --port 8000
```

When you see the line output log: NGROK TUNNEL URL: https://xxxx-xxxx.ngrok-free.app, copy that link to use in Step 3. This cell will keep running to maintain the server.

### Step 3: Run the Web UI on Your Local Machine
Once the Kaggle Server is ready and waiting for requests, switch back to your local computer:

Open `client.py`

Paste the Ngrok link you copied in Step 2 into the corresponding variable:

```Python
KAGGLE_NGROK_URL = "https://xxxx-xxxx.ngrok-free.app"
```
Open a terminal on your local machine and start the Gateway:

```Bash
python client.py
```

Open your browser and navigate to `http://127.0.0.1:8080` to experience the app powered by Kaggle's GPU.