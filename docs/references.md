# References

## Colton Downing's submissions

**Downing, Colton. “Blend 34 ours 66 public CNN 30.csv.” UMUD Challenge, Kaggle, 11 November 2026. Public score: 0.23551.** [Team submission history](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/submissions).

This is the reference used in both submitted hybrids. It is included as [colton_blend34.csv](../references/colton_blend34.csv), byte-identical to the supplied `BEST submission 0.23551.csv`.

SHA256: `7f155ec03ac1d5b94353039ded4c9a9b00f1022806d0fc00aad6bc0926eb3ca6`.

Colton also supplied `build_best.py`, `our_pipeline_G7.csv`, `public_pipeline_0.30108.csv`, `cnn_average_20_models.csv`, and `scanner_family.csv`. His script blends 34% G7 with 66% public predictions, applies a 30% scanner-median-aligned CNN component to angle and length, and adds 1° to angle. It reproduces the reference's 927 target values. The underlying G7 and CNN training source was not provided; this repository does not claim authorship of those components or identify the public CSV's author without evidence.

Earlier Colton submissions recorded in the same team history include `submission_blend_g7_pub30108_cnn.csv` (0.24640) and `Blend 38 ours 62 public CNN 30.csv` (0.23558). These are comparison results, not inputs to this repository's hybrids. Individual public notebook links were not attached to the CSV submissions.

## Methods and data

- ambrosm. [UMUD Quick and Dirty, version 1](https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty). Source of the known-display ruler rules adapted in `calibration.py`; the notebook’s prediction formula was not used for the attention measurements.
- Oktay, O., et al. (2018). [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999). Additive attention-gate design.
- [Torchvision ResNet34](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html). Encoder implementation and ImageNet weights.
- Shit, S., et al. (2021). [clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/abs/2003.07311). Reference for the optional topology loss; its weight was zero in the submitted models.
- Ritsche, P. [UMUD Challenge data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data) and [UMUD score, version 9](https://www.kaggle.com/code/paulritsche/umud-score?scriptVersionId=307798454). Competition images, masks, and evaluation formula.
- [DL_Track_US](https://github.com/PaulRitsche/DL_Track_US). Earlier project baseline and public example-data reference; its pretrained models were not the Attention U-Net checkpoints used here.
