# A–I loss controls

Let `z` denote the `K` class logits, `e` the one-hot label for class `y`, `v` a separate `K`-output linear head, and `s` a separate head containing 50 blocks of five outputs. `T[y]` is the fixed five-way target for class `y`, and `mean(T)` is its equal-class average. MSE averages over all target coordinates. Every arm predicts with `argmax(z)`.

| Arm | CLI name | Objective |
| --- | --- | --- |
| A | `ce` | Class cross-entropy |
| B | `raw-onehot` | MSE of raw class logits `z` against `e` |
| C | `prob-onehot` | MSE of softmax of `z` against `e` |
| D | `ce-same-onehot` | Class cross-entropy plus β times the MSE of `z` against `e` |
| E | `squentropy` | Class cross-entropy plus the mean squared raw logit over the `K − 1` incorrect classes, with coefficient 1 |
| F | `ce-independent-onehot` | Class cross-entropy plus β times the MSE of `v` against `e`. The initial parameters of `v` are copied from the classification head. |
| G | `ecoc-soft-ce` | Class cross-entropy plus β times the mean soft-target cross-entropy between the five-way blocks of `s` and `T[y]` |
| H | `ecoc` | Class cross-entropy plus β times the MSE of raw auxiliary outputs `s` against `T[y]` |
| I | `ecoc-constant` | Class cross-entropy plus β times the MSE of raw auxiliary outputs `s` against `mean(T)` |

The historical settings use β = 50 on Flowers and β = 40 on CUB, including arms D and F. The code symbols for G, H, and I are generated from integer class IDs. Softening around the uniform distribution gives them one aggregate class-centered energy from the published Qwen probabilities. The separate ResNet-18 `ecoc-hard` arm uses unsoftened one-hot five-way targets and can be selected directly with `ecg_lavid.train_visual_lavid`. G★ scales G to match H's auxiliary gradient norm on the shared input layer of the first training batch, then holds the scale fixed. [Squentropy](https://proceedings.mlr.press/v202/hui23a.html)

## Image experiments

Follow the [shared setup](../SETUP.md) including the pinned LaViD checkout. Set the image root to the directory containing the files below.

| Dataset | Required layout | Train / held out |
| --- | --- | --- |
| `flowers` | `jpg/`, `imagelabels.mat`, `setid.mat`. The author code combines official train and validation for training. | 2,040 / 6,149. [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| `cub200` | `images/`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `classes.txt`. | 5,994 / 5,794. [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) |

The LaViD paper appendix describes a different Flowers split of 6,552/1,637. The fixed final epoch is the main comparison here. Selection by lowest class cross-entropy on the same evaluation split is recorded separately for comparison with the author code. [Author data split code](https://github.com/lliangthomas/lavid/blob/fa689286d168eb7daa9899239461cfffc5517de1/utils.py#L44-L97)

The [visual launcher](run_visual.sh) uses seed 17 and launches completed conditions. That includes A–I and G★ for both CNNs, and A, B, C, E, and G at the original weights for ViT-S/16. The CNNs use 240 epochs and batch size 16. ViT-S/16 uses 500 epochs, physical batch size 64, and effective batch size 512 with cosine scheduling. Supply the ViT weights file as the last argument. Its SHA-256 is `79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa`. [Pinned ViT-S/16 weights](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/7e2c55630205e1266030f18370f4c6ed1a514b52/model.safetensors)

```bash
bash experiments/loss-controls/run_visual.sh flowers resnet18 \
  data/flowers102 lavid-upstream runs/loss-resnet18
bash experiments/loss-controls/run_visual.sh cub200 mobilenetv2 \
  data/CUB_200_2011 lavid-upstream runs/loss-mobilenetv2
bash experiments/loss-controls/run_visual.sh flowers vit_small_patch16 \
  data/flowers102 lavid-upstream runs/loss-vit data/vit-small.safetensors
```

For individual arms, use `ecg_lavid.train_visual_lavid` for ResNet-18 and `ecg_lavid.train_visual_loss_matrix` for MobileNetV2 and ViT-S/16. For MobileNetV2 G★, run `--calibrate-gradmatch --gradient-calibration <calibration.json>` once before selecting `--arms ecoc-soft-ce-gradmatch --gradient-calibration <calibration.json>`.

### ViT-S/16 G and H at the ResNet-18 H gradient ratio

In this separate completed comparison, both losses target the ratio observed in seed-17 ResNet-18 H on the same dataset. The anchor is the first augmented training batch, of size 16, at the shared input convolution. Use the CE and unweighted auxiliary gradient norms captured by the ResNet-18 H training history to form `β × norm(aux) / norm(CE)`. The ViT launcher measures the respective unweighted ratio on its own first effective batch of 512 at the patch projection, divides the ResNet-18 H target by that ratio, and holds the resulting coefficient fixed. It records the first effective batch at epochs 1–10 and every tenth epoch thereafter.

| Dataset | ResNet-18 H target ratio | Historical matched G β | Historical matched H β |
| --- | ---: | ---: | ---: |
| Flowers-102 | 2.8399443254 | 15.0212079459 | 5.0223133842 |
| CUB-200-2011 | 2.4939642847 | 19.0170570889 | 7.4914678235 |

The original G★ uses H's auxiliary gradient on the same backbone as its anchor. Use the separate output branch below for the ResNet-18 H target on ViT-S/16. The launcher reproduces the gradient calculation from the frozen initialization and regenerated targets.

```bash
bash experiments/loss-controls/run_visual.sh --match-resnet18-h flowers vit_small_patch16 \
  data/flowers102 lavid-upstream runs/loss-vit data/vit-small.safetensors
```

The [Waterbirds-95 protocol and launcher](WATERBIRDS.md) use the official three-way split and report each of the four bird/background groups.

### Paired image interventions

Evaluate each arm on a common cohort with official segmentation images. The evaluator freezes image and donor IDs and implements nine views including grayscale, isolated foreground or background, background replacement, and matched controls. Reuse the same manifest for each arm. Supply `--architecture mobilenetv2` or `--architecture vit_small_patch16` when evaluating a model from the matrix launcher.

```bash
python -m ecg_lavid.visual_interventions --dataset flowers --images data/flowers102 \
  --masks data/flowers-segmentations --manifest runs/flowers-mask-cohort.json \
  --author-source lavid-upstream --checkpoint runs/loss-resnet18/flowers/seed-17/ce/final.pt \
  --out runs/flowers-ce-interventions.npz
```

Use `--manifest-only` with the dataset, image and segmentation paths to generate the shared cohort before scoring any checkpoints.

To reconstruct the comparison from individual predictions, pass the CE file and one or more named arms. The summary checks that image IDs, ground-truth labels and background donors match, then reports correct counts and the percentage-point difference from CE for each view.

```bash
python experiments/loss-controls/summarize_masks.py --ce runs/flowers-ce-interventions.npz \
  --arm H runs/flowers-ecoc-interventions.npz --out runs/flowers-nine-views.csv
```

The [true-class Grad-CAM analysis](gradcam/) compares these same model pairs on the same images and measures foreground activation relative to foreground area.

## PTB-XL waveform controls

Prepare `runs/ecg/ptbxl-rhythm.npz` using the [ECG launcher](../ecg-distillation/). The waveform experiment uses 12 five-way blocks, a fixed label-only code, and β = 0.4. The CNN and BiLSTM trainer implements A–H and G★. The [waveform launcher](run_waveform.sh) trains both backbones at the complete patient budget and seed 17, then evaluates the validation-selected checkpoints for A–G and G★ on fold 10 with paired missing-waveform windows. Set `MISSING=interpolation` to evaluate interpolated windows. `ecg_lavid.ptb_crossmodal_eval` evaluates the original H arm separately.

```bash
bash experiments/loss-controls/run_waveform.sh runs/ecg/ptbxl-rhythm.npz runs/waveform-losses
```
