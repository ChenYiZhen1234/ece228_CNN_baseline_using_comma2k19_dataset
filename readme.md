# ECE228 Project — Baseline CNN for Speed & Steering Prediction

This is our baseline model for the ECE228 course project. The goal is to predict vehicle **speed** and **steering angle** using dashcam images and CAN bus data from the [comma2k19](https://github.com/commaai/comma2k19) dataset.

We kept the architecture simple on purpose — this is meant to be the benchmark that we (hopefully) beat with better models later.

---

## What does this do?

Instead of just feeding in a single frame, we stack the **previous 3 frames** together as a 9-channel input so the model can pick up on motion (e.g. the car is turning, speeding up, etc.). We also feed in the CAN readings from those same 3 timesteps.

The model then predicts what the speed and steering angle will be **1 second in the future**.

```
prev 3 frames (64×64, stacked) ──┐
                                  ├──► CNN + MLP ──► [speed (m/s), steer (deg)]
prev 3 CAN readings ─────────────┘
```

---

## Model

Nothing fancy — a small CNN followed by an MLP that fuses the image features with the CAN data.

**CNN:**
- Input: 9-channel image (3 RGB frames concatenated along channel dim)
- 4 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool)
- Channels: 16 → 32 → 64 → 128

**MLP:**
- Flattened CNN output (2048-dim) + CAN vector (6-dim: speed×3 + steer×3)
- Linear(2054 → 256) → ReLU → Dropout(0.3) → Linear(256 → 64) → ReLU → Linear(64 → 2)
- Outputs: `[speed_pred, steer_pred]`

---

## Dataset

We used **comma2k19 Chunk_1**. The dataset contains dashcam footage + synced CAN bus data from real highway drives.

- Train / Val split: 80% / 20% by segment
- Image size resized to 64×64 (easier to run locally on CPU)
- Prediction target: vehicle state 1 second ahead

Download the dataset [here](https://github.com/commaai/comma2k19) and place it at:
```
data_utils/comma2k19_data/extracted/Chunk_1/
```

---

## Training Setup

| Setting | Value |
|--------|-------|
| Image size | 64×64 |
| Batch size | 16 |
| Epochs | 10 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| LR schedule | StepLR (×0.5 every 5 epochs) |
| Loss | Normalised MSE (divided by 30 for both outputs) |

We normalise the loss because speed (0–35 m/s) and steering (−30 to +30 deg) are on different scales — without this, the loss would be dominated by whichever has larger values.

---

## How to run

**1. Install dependencies**
```bash
pip install torch torchvision opencv-python matplotlib tqdm
```

**2. Make sure `data_loader.py` is in the same folder as the notebook**

**3. Open the notebook and run all cells**
```
baseline_training_local_previous3frame.ipynb
```

If you just want to quickly check that everything runs without waiting forever, set `QUICK_TEST = True` in the Config cell — it only uses the first 5 segments and finishes in a few minutes.

> **Windows users:** keep `NUM_WORKERS = 0` or you'll get multiprocessing errors.

---

## Output

After training you'll find these files in the same directory:

| File | What it is |
|------|------------|
| `baseline_best.pth` | Best model checkpoint (saved automatically) |
| `history.json` | Loss + MAE for every epoch |
| `baseline_training_curves.png` | Training curves plot |

The final cell also plots predicted vs. ground truth speed and steering for the first 175 validation frames so you can visually see how well it's doing.

---

## Baseline Results

The notebook prints a summary at the end:

```
=============================================
       BASELINE BENCHMARK RESULTS
=============================================
  Best epoch       : X / 10
  Val loss (norm)  : X.XXXX
  Speed MAE        : X.XXX m/s
  Steering MAE     : X.XXX deg
=============================================
  ← This is your benchmark to beat!
```

---

## File Structure

```
plus_CAN/
├── baseline_training_local_previous3frame.ipynb   # main notebook
├── data_loader.py                                  # loads comma2k19 segments
├── baseline_best.pth                               # best model weights (generated)
├── history.json                                    # training log (generated)
└── baseline_training_curves.png                    # loss curves plot (generated)
```
