# ViT-B/16 CE and auxiliary losses

All arms fine-tune TorchVision ViT-B/16 from ImageNet-1K V1 weights at seed 17. The Qwen arm adds MSE from a separate 250-output head to LaViD's published Qwen2.5-7B probabilities for the image class. All arms predict from the ordinary classification head. [Input checksums and run settings](provenance.json) · [Pinned LaViD release](https://github.com/lliangthomas/lavid/tree/fa689286d168eb7daa9899239461cfffc5517de1)

## Reproduce the pair

Install the [shared dependencies](../../SETUP.md) and use a CUDA GPU with BF16 support. The reference run used PyTorch 2.8.0 and TorchVision 0.23.0 with CUDA 12.8. The trainer downloads and checks the official weights in the Torch Hub cache on first use. [TorchVision weights](https://download.pytorch.org/models/vit_b_16-c867db91.pth)

Use the [benchmark dataset layout](../README.md#other-students-and-datasets). Flowers combines Oxford's official training and validation folds and holds out the test fold (2,040/6,149 images). CUB uses its official train/test split (5,994/5,794). Run the commands from the repository root. Each Qwen arm uses the recorded first batch of its CE pair in the same output root. On Windows, append `--workers 0` if needed.

```bash
python experiments/visual-distillation/vit-b16/train.py --dataset flowers --arm ce --images data/flowers102 --out runs/vit-b16
python experiments/visual-distillation/vit-b16/train.py --dataset flowers --arm qwen --images data/flowers102 --author-source lavid-upstream --out runs/vit-b16

python experiments/visual-distillation/vit-b16/train.py --dataset cub200 --arm ce --images data/CUB_200_2011 --out runs/vit-b16
python experiments/visual-distillation/vit-b16/train.py --dataset cub200 --arm qwen --images data/CUB_200_2011 --author-source lavid-upstream --out runs/vit-b16
```

The default batch size is 32 with accumulation to 512 images. SGD starts at 0.01 with momentum 0.9 and weight decay 0.0005. The learning rate follows a 500-epoch cosine schedule. Before the first optimizer update, Qwen calibrates a fixed coefficient from the ratio of MSE to CE gradient norms on the ViT patch projection. It matches the dataset's ResNet-18 H reference ratio of 2.8399443 for Flowers and 2.4939643 for CUB. The resulting coefficients were 12.8827686 and 11.3801812. The script checks the shared initialization and augmented first batch.

Progress is saved under `<out>/<dataset>/vit_b_16/seed-17/<arm>/`. Repeat the same command to resume or use `--max-new-epochs` to bound one invocation. The default evaluation loads the original images. To build a pixel-checked cache, run `python -m ecg_lavid.visual_validation_cache --dataset flowers --images data/flowers102 --cache runs/flowers-eval.npy`, then add `--validation-cache runs/flowers-eval.npy` to each Flowers command.

## Flowers-102 five-way class-code control

After both Flowers runs finish, launch [H](train_h.py) against the same output root with the same batch size and evaluation reader.

```bash
python experiments/visual-distillation/vit-b16/train_h.py --images data/flowers102 --author-source lavid-upstream --out runs/vit-b16
```

H uses CE plus MSE to a fixed 50×5 class code from the [shared generator](../../../ecg_lavid/visual_benchmark_data.py) at code seed 271828. Rows are assigned by class index. A single global softening scale of 0.6754907644 matches the class-centered energy of the published Flowers Qwen signatures. The script checks the target matrix hash and calibrates the fixed MSE coefficient to the same ResNet-18 H first-batch stem gradient ratio used for Qwen. The reference ViT-B/16 run obtained a weight of 12.9813618 and a ratio of 2.8399443. It stores H under `flowers/vit_b_16/seed-17/ecoc/` and verifies the paired schedule and initialization after completion.

## Reference accuracy

The fixed checkpoint uses epoch 500. The selected checkpoint minimizes classification NLL among epochs 5, 10, …, 500 on the held-out split. The table shows FP32 Top-1 on that same split.

| Dataset | CE fixed | Qwen fixed | CE selected | Qwen selected |
| --- | ---: | ---: | ---: | ---: |
| Flowers-102 | 96.9914% | 95.9343% | 97.1865% (385) | 95.9180% (375) |
| CUB-200-2011 | 81.3255% | 81.4808% | 82.2402% (60) | 82.3611% (75) |

Flowers H reached 95.3000% at epoch 500 and 95.3976% at the selected epoch 425.
