# UMUD attention and geometry

Separate ResNet34 Attention U-Nets segment aponeuroses and fascicles in ultrasound images. The geometry code estimates pennation angle, fascicle length, and muscle thickness from the predicted masks.

The submitted model builds on [Colton Downing's 0.23551 submission](docs/references.md). It uses 20% attention-model measurements and 80% of Colton's predictions on 102 images that pass the geometry checks. The other 207 images keep his predictions. The best public score is **0.23276**.

| Submission | Public score ↓ |
|---|---:|
| [Colton's reference](references/colton_blend34.csv) | 0.23551 |
| [Attention + geometry, 20%](submissions/attention_hybrid20.csv) | **0.23276** |
| [Trimmed fascicle-length variant](submissions/attention_hybrid20_trimmed_fl.csv) | 0.23280 |

## Run

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest tests umud_attention/test_geometry.py
python scripts/reproduce_submissions.py
```

Use [the notebook](notebooks/UMUD_Attention_UNet.ipynb) for data preparation, training, and inference. Competition access and data must be obtained through [Kaggle](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data). Model weights and ultrasound data are not stored in Git. The reconstruction script verifies the first submission numerically and reproduces the second byte for byte from the saved measurements and segment lengths.

To train fold 0 from extracted competition data:

```bash
python scripts/train_competition.py --data-root /path/to/data --output-dir runs/fold0
```

The main files are [model.py](umud_attention/model.py), [train.py](umud_attention/train.py), [geometry.py](umud_attention/geometry.py), and [infer.py](umud_attention/infer.py). See [results](docs/results.md) for training and validation details, and [references](docs/references.md) for attribution.

Physical calibration remains experimental. These scores describe the hybrid with Colton's predictions, not standalone Attention U-Net performance.
