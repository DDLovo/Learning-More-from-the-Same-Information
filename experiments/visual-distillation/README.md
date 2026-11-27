# Visual distillation benchmark

The hybrid trains a randomly initialized vision student with the author's two-view MLKD objective and logit standardization, plus a weighted MSE on an independent projection of the student's weak-view features. A fixed supervised ResNet-50 supplies weak- and strong-view logits to MLKD. The auxiliary MSE uses either published Qwen2.5-7B signatures or a fixed five-way class code softened to match the Qwen signatures' aggregate class-centered energy. A separate code-only arm uses single-view cross-entropy plus the code MSE without the vision teacher. Each method predicts with its classification head. [LaViD paper](https://proceedings.mlr.press/v306/liang26ac.html) · [A–I target definitions](../loss-controls/)

## Flowers-102 ResNet-18 reproduction

Follow the [shared setup](../SETUP.md) including the pinned LaViD checkout. Put `jpg/`, `imagelabels.mat`, and `setid.mat` in `data/flowers102`. The [Flowers launcher](run_flowers.sh) builds the historical code, trains a ResNet-50 teacher at seed 17, and uses the checkpoint selected by held-out classification loss for all students. It runs the three Qwen hybrids first, then their code-hybrid and code-only comparisons at seeds 17, 29, and 43. The paired hybrids share the teacher checkpoint, initialization, and augmentation schedule. The CNN schedule is 240 epochs with batch size 16 and SGD at learning rate 0.01, momentum 0.9, and weight decay 0.0005. Learning rates drop by a factor of 0.1 at epochs 150, 180, and 210. [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

```bash
bash experiments/visual-distillation/run_flowers.sh data/flowers102 lavid-upstream runs/flowers-distillation
```

To cache original-size RGB training images, run `python experiments/visual-distillation/build_flowers_rgb_cache.py --images data/flowers102 --out runs/flowers-rgb-cache`, then set `FLOWERS_RGB_CACHE=runs/flowers-rgb-cache` before launching training. To reuse the two native BF16 vision-teacher outputs, set `TEACHER_LOGIT_CACHE=runs/flowers-teacher-logits`. The Qwen hybrids record the logits and the paired code hybrids replay them. The original direct trainer commands remain available as `ecg_lavid.train_visual_lavid_teacher`, `ecg_lavid.train_visual_lavid_ls`, and `ecg_lavid.train_visual_benchmark`.

## Other students and datasets

The [benchmark launcher](run_benchmark.sh) takes a dataset, student architecture, image root, source checkout, output directory, and student seed. It generates or reuses the common cohort and teacher, then runs `lavid-ls-qwen`, `lavid-ls-ecoc`, and `ecoc-only`. Repeat with seeds 17, 29, and 43 under the same output directory to compare paired runs. Valid student architectures are `resnet18`, `mobilenetv2`, and `shufflenetv2` (x1.0).

```bash
bash experiments/visual-distillation/run_benchmark.sh cub200 mobilenetv2 \
  data/CUB_200_2011 lavid-upstream runs/visual-grid 17
```

| CLI dataset | Required layout and train/test size |
| --- | --- |
| `flowers` | Oxford image and MAT files above. 2,040 / 6,149. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| `cub200` | Official root with `images/`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, and `classes.txt`. 5,994 / 5,794. [Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/) |
| `fgvc_aircraft` | TorchVision layout with `fgvc-aircraft-2013b/data/` and `images/`. Variant-level trainval/test. 6,667 / 3,333. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) |
| `oxfordpets` | TorchVision layout under `oxford-iiit-pet/` with images and annotations. Trainval/test. 3,680 / 3,669. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) |
| `stanfordcars` | TorchVision layout under `stanford_cars/` with both splits, devkit, and test labels. 8,144 / 8,041. [TorchVision recovery information](https://github.com/pytorch/vision/issues/7545) |
| `caltech101` | Root with `101_ObjectCategories/`. Uses the author's split seed 27 after sorting files within each class. Both face labels are retained and share a language signature. 4,310 / 4,367. [Dataset](https://data.caltech.edu/records/mzrjq-6wc02) |

The launcher uses `$AUTHOR_SOURCE/questions/<dataset>/logits/qwen2.5-7b_chat_50-5.json` for the first five datasets. For Aircraft, Pets, Cars, and Caltech it also invokes [the cohort prep script](prepare_benchmark_cohort.py). Its `cohort.csv` and `cohort.json` sidecar preserve the split and label order for code generation, teacher training, and student training. The author code uses a single held-out split for evaluation and checkpoint selection, including 6,149 Flowers images. The paper appendix gives a Flowers split of 6,552/1,637. [Author data split code](https://github.com/lliangthomas/lavid/blob/fa689286d168eb7daa9899239461cfffc5517de1/utils.py#L44-L97)

### Caltech-101 7B target

The author's released Caltech target is labeled 72B. The [reconstruction entry point](reconstruct_caltech7b.py) runs the author's scorer from a local model snapshot. A Flowers subset validates the scorer against the published 7B scores. Use the `score` command first with `--dataset flowers`, inspect the calibration audit, then rerun with `--dataset caltech101` and a separate output file. The accepted model IDs and revisions are listed below. [Published Caltech scores](https://github.com/lliangthomas/lavid/blob/fa689286d168eb7daa9899239461cfffc5517de1/questions/caltech101/logits/qwen2.5-72B_chat_50-5.json)

| Model ID | Revision |
| --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` |

```bash
python experiments/visual-distillation/reconstruct_caltech7b.py score --author-root lavid-upstream --dataset flowers \
  --model-id Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --model-path data/qwen7b-instruct --out runs/flowers-qwen-calibration.json
python experiments/visual-distillation/reconstruct_caltech7b.py score --author-root lavid-upstream --dataset caltech101 \
  --model-id Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --model-path data/qwen7b-instruct --out runs/caltech-qwen.json
bash experiments/visual-distillation/run_benchmark.sh caltech101 resnet18 \
  data/caltech101 lavid-upstream runs/visual-grid 17 runs/caltech-qwen.json
```

The example uses the Instruct snapshot, stored under `data/qwen7b-instruct`. Preserve the candidate's provenance and Flowers calibration scores for comparisons with the paper.

### Individual benchmark arms

Use `ecg_lavid.train_visual_benchmark --method lavid-ls-qwen` and then `--method lavid-ls-ecoc` under the same output root for a paired hybrid. Provide the same `--teacher-checkpoint` and change `--target` between the Qwen signature and generated code. `ecoc-only` takes the code and runs without a visual checkpoint. If caching, pass `--teacher-logit-cache` to both hybrid runs and add `--record-teacher-logits` to the Qwen run.
