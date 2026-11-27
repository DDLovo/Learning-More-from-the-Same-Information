# True-class Grad-CAM on Flowers and CUB

This analysis compares the final convolutional representation of a cross-entropy model (A) with that of a model trained with class codes (H). Both maps use the logit of the ground-truth species, including when a model predicts the wrong class. The completed comparisons use ResNet-18 and MobileNetV2, seed 17, and the checkpoint at epoch 240.

First run the [nine-view intervention evaluator](../README.md#paired-image-interventions) on each checkpoint with the same segmentation manifest. Keep the resulting prediction NPZ files and their companion JSON files together. Grad-CAM uses four of those nine views and checks the image IDs, donors, masks and checkpoint hashes against the predictions.

From the repository root, the following example uses the Flowers ResNet-18 pair. Set `A_CHECKPOINT` and `H_CHECKPOINT` to the respective `seed-17/<method>/final.pt` files.

```bash
python -m ecg_lavid.visual_gradcam --dataset flowers \
  --images data/flowers102 --masks data/flowers-segmentations \
  --manifest runs/flowers-mask-cohort.json --author-source lavid-upstream \
  --model resnet18 A "$A_CHECKPOINT" runs/flowers-ce-interventions.npz \
  --model resnet18 H "$H_CHECKPOINT" runs/flowers-ecoc-interventions.npz \
  --examples experiments/loss-controls/gradcam/display-ids.json --out runs/cam

python experiments/loss-controls/gradcam/summarize.py --inputs runs/cam \
  --dataset flowers --architecture resnet18 --out runs/cam/flowers-r18.csv

# Optional plotting dependency: python -m pip install matplotlib
python experiments/loss-controls/gradcam/render.py \
  --pair runs/cam/flowers/resnet18 --out runs/cam/figures/flowers/resnet18
```

The summary covers all 5,978 eligible Flowers images or all 5,794 CUB test images. For each image, it divides the fraction of positive Grad-CAM mass in the foreground by the foreground area fraction. A score of 1 corresponds to uniform mass per pixel. On background-only inputs, the mask marks the original footprint of the removed subject. The CSV reports paired H–A differences with 10,000 paired image bootstrap draws, along with counts of zero or undefined maps. These intervals describe variation across images for the fixed model pair.

The optional display IDs reproduce four classification-outcome examples per dataset. They are shared across architectures. The renderer uses one color scale per image across both models and all three displayed views. The full-cohort CSV estimates the average difference in foreground activation across paired images.
