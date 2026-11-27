# Waterbirds-95 loss controls

Use the official `waterbird_complete95_forest2water2` archive. The image root contains `metadata.csv` and the numbered CUB species directories. Metadata fixes the 4,795 training images, 1,199 validation images and 5,794 test images through split IDs 0, 1 and 2. The loader uses the original CUB image ID from the metadata. [Group DRO data and splits](https://github.com/kohpangwei/group_DRO#waterbirds) · [LaViD dataset implementation](https://github.com/lliangthomas/lavid/blob/fa689286d168eb7daa9899239461cfffc5517de1/lavid/data/custom_loaders.py#L90-L129)

The launcher's completed subset uses seed 17, the author ResNet-18 or MobileNetV2, and the same image transforms, 240 epochs, batch size 16, and SGD schedule as the other CNN loss controls. The frozen Waterbirds coefficient is β=40 for weighted arms. E uses its original coefficient of 1.

| Backbone | Completed arms selected by the launcher |
| --- | --- |
| ResNet-18 | A, B, C, E, G |
| MobileNetV2 | A, B, G |

Every method predicts landbird or waterbird. A–F use these two labels in their class losses. G and H look up a fixed 50×5 probability code by the 200-way CUB species index from the image path. I uses the equal-species code mean. G★, when available, matches G's first-batch auxiliary gradient norm to H's on the same backbone. The launcher regenerates the same CUB code as the main loss controls. Its global softening scale derives from the published Qwen target matrix. Each artificial code row is assigned by the species index.

```bash
bash experiments/loss-controls/run_waterbirds.sh resnet18 \
  data/waterbird_complete95_forest2water2 lavid-upstream runs/waterbirds
bash experiments/loss-controls/run_waterbirds.sh mobilenetv2 \
  data/waterbird_complete95_forest2water2 lavid-upstream runs/waterbirds
```

Results include the fixed epoch-240 checkpoint and the checkpoint with the lowest classification cross-entropy on the official validation split. They score both checkpoints on the separate test split. The per-image archives contain IDs, two-class log probabilities, the bird and background labels and the fine species ID. Results report image-weighted overall accuracy, the equally weighted mean of the four group accuracies, the minimum group accuracy, and the accuracy reweighted to the training group proportions.

| Group ID | Bird label | Background |
| ---: | --- | --- |
| 0 | Landbird | Land |
| 1 | Landbird | Water |
| 2 | Waterbird | Land |
| 3 | Waterbird | Water |
