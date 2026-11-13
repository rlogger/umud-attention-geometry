# Results

## Training

Two ImageNet-pretrained ResNet34 encoders, each with four additive skip-attention gates. Both models used real competition images and supplied masks, fold 0, AdamW, learning rate 0.0002, batch size 8, and BCE + Dice loss with auxiliary supervision. The optional clDice loss was disabled.

| Model | Resolution | Training / validation rows | Epochs | Best Dice |
|---|---:|---:|---:|---:|
| Aponeurosis | 512 × 512 | 698 / 350 | 40 | 0.8273 |
| Fascicle | 768 × 768 | 1,838 / 918 | 36 | 0.2870 |

Counts are task-specific annotation rows, not independent subjects. Exact duplicates and horizontal-flip families were grouped. Subject independence and other near-duplicate separation remain unverified.

## Measurements

Inference averages original and horizontally flipped predictions, restores masks to image coordinates, and fits muscle boundaries and fascicle directions. Pennation angle uses the local deep-boundary tangent. Fascicle length uses the distance between intersections with the two boundaries. Thickness uses normal distances over the central boundary region.

Central probability thresholds are 0.65 for aponeuroses and 0.35 for fascicles, with offsets −0.08, 0, and +0.08. Blending requires at least two accepted fascicle estimates and stable measurements across at least two threshold variants. Of 309 test images, 102 passed; 207 retained Colton's reference.

## Submitted comparisons

Both submissions completed on 12 November 2026. [Kaggle submission history](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/submissions) is visible to team members.

| File in this repo | Public score ↓ |
|---|---:|
| `references/colton_blend34.csv` | 0.23551 |
| `submissions/attention_hybrid20.csv` | **0.23276** |
| `submissions/attention_hybrid20_trimmed_fl.csv` | 0.23280 |

The first hybrid reduced the reference score by 0.00275, or 1.17%. The second changed only fascicle-length aggregation: a mean after filtering values beyond three scaled median absolute deviations, followed by the median across thresholds. PA, MT, the blend weight, and the 102 eligible images stayed fixed.

On the same 29 paired held-out images, the trimmed version reduced mask-derived FL mean absolute error from 34.09 to 25.12 pixels: 20 cases improved, 6 worsened, and 3 tied. The improvement did not carry over to Kaggle; the first hybrid remains better by 0.00004.

## Limits

The 29-case references came from annotation masks using the same geometry code, not independent expert measurements. All lacked recognized physical calibration. Test-image ruler templates assume square pixels without independent verification, and full fascicle lengths can require substantial extrapolation. The lower fascicle Dice remains a limitation.

The validation run passed 151 software tests and 36 real-map invariance checks. These check implementation behavior, not anatomical accuracy. The public score does not establish a private-leaderboard result or a standalone attention-model score. Reproducing the full hybrid on new images also requires Colton's prediction pipeline.

[Training record](../results/training.json) · [Validation statistics](../results/validation.json) · [Submission hashes](../results/scores.json) · [Export checks](../results/export_checks.json)
