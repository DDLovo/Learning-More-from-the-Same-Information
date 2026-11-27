# Learning More from the Same Information

This repository contains training and evaluation code for studying whether additional objectives help a model learn from a fixed set of class labels. The experiments compare ordinary cross-entropy with redundant class codes and language-model signatures. In visual–language distillation experiments, a separate supervised vision teacher provides another source of training signal. Students predict with their ordinary classification heads from waveforms or images.

1. [ECG language distillation](#1-ecg-language-distillation) trains a waveform classifier from Qwen and Muse Spark Open signatures on PTB-XL.
2. [A–I loss controls](#2-ai-loss-controls) compare target encodings, output heads, and loss functions on Flowers-102 and CUB-200-2011, with complementary ECG controls.
3. [Visual distillation benchmark](#3-visual-distillation-benchmark) combines a ResNet-50 teacher with Qwen or five-way codes, and runs five-way-code supervision on its own.

## Setup

Use Python 3.11 or newer and a matched PyTorch and TorchVision build. Full visual training and the pinned language-model scoring runs use CUDA with BF16 support. Run the commands from the repository root. The examples use Bash syntax. Use `--workers 0` for the visual hybrid trainers on Windows.

```bash
python -m pip install -r requirements.txt
mkdir -p data runs
OUT=runs
```

For the visual experiments, obtain the upstream source and published targets at the pinned LaViD commit. Disabling Git's line-ending conversion preserves the byte hashes checked by the code on Windows.

```bash
git -c core.autocrlf=false clone https://github.com/lliangthomas/lavid.git lavid-upstream
git -C lavid-upstream -c core.autocrlf=false checkout --detach fa689286d168eb7daa9899239461cfffc5517de1
AUTHOR=lavid-upstream
```

The visual code uses the public source checkout's model definitions, augmentation, and MLKD objective. [LaViD paper](https://proceedings.mlr.press/v306/liang26ac.html) · [Pinned author source](https://github.com/lliangthomas/lavid/tree/fa689286d168eb7daa9899239461cfffc5517de1)

## 1. ECG language distillation

The ECG task uses 10-second, 12-lead, 100 Hz PTB-XL v1.0.3 recordings with one annotated rhythm in the four-class cohort `SR`, `AFIB`, `STACH`, and `SBRAD`. Official folds 1–8 train the student, fold 9 selects the checkpoint by macro-F1, and fold 10 provides the held-out test. The low-label setting subsamples training patients and fits lead normalization on that subset. Download the waveform release into `data/ptb-xl-1.0.3`, with `ptbxl_database.csv`, `scp_statements.csv`, and `records100/` in that directory. Alternatively, add `--download-workers 16` to fetch the metadata and the waveform files used by the cohort. [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

```bash
python -m ecg_lavid.prepare --ptbxl data/ptb-xl-1.0.3 --out "$OUT/ptbxl-rhythm.npz"
python -m ecg_lavid.rhythm_question_bank --bank astra \
  --out "$OUT/ecg-questions.json" --rubric "$OUT/ecg-rubric.json"
```

The fixed question bank asks about observable ECG concepts such as ventricular rate, R–R intervals, atrial activity, and P–QRS timing. The language teachers receive only class names and the question text. The trainer assigns the resulting class-by-question-by-option probabilities using the true class label. A separate linear head learns the target with mean squared error, and class cross-entropy uses the original classification head. The `permuted` arm reassigns whole signatures with a fixed class derangement.

### Score with Qwen

Qwen scores the five answer letters at the assistant's first token using its native chat template. The code normalizes the five logits with a temperature of one. The earlier scoring control uses `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The option `--bank manual` materializes the original manually prepared question bank. To generate a fresh bank, use `ecg_lavid.teacher` without `--questions`, or use `ecg_lavid.generate_openai` with a supported model and `OPENAI_API_KEY`. Freshly generated banks define a different run. [Pinned Qwen3 weights](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) · [Pinned Qwen2.5 weights](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

```bash
python -m ecg_lavid.teacher --model Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --questions "$OUT/ecg-questions.json" --out "$OUT/ecg-qwen.json" --device cuda --dtype bfloat16
python -m ecg_lavid.train --data "$OUT/ptbxl-rhythm.npz" --teacher "$OUT/ecg-qwen.json" \
  --out "$OUT/ecg-qwen-students" --device cuda --methods ce lavid permuted \
  --seeds 17 29 43 --epochs 25 --fractions 0.1 --lavid-weight 0.4 --validation-only
```

The example uses the 10%-patient validation protocol. Remove `--validation-only` and use a fresh output folder for the train/validation/test run. The training configuration records the target fingerprint, the patient subset, the seed that controls batch order, the schedule, and the class permutation.

### Score with Muse Spark Open

Provide an OpenRouter credential through `OPENROUTER_API_KEY`. The scorer requests `meta/muse-spark-open-20261015` on `nebius/fp4` with top-20 token log-probabilities. Rerunning the API step depends on the availability of that route and log-probability support. The audit locates the first visible token after any returned internal tokens. If some bare answer letters fall outside a verified top-20 list, the generated signature uses half the cutoff for each missing option, then renormalizes over A–E. Those rows are estimates, and the audit reports their missing probability mass bounds. [OpenRouter model page](https://openrouter.ai/meta/muse-spark-open)

```bash
python -m ecg_lavid.score_muse_concepts --questions "$OUT/ecg-questions.json" \
  --rubric "$OUT/ecg-rubric.json" --out "$OUT/ecg-muse" --cache "$OUT/ecg-muse-cache"
python -m ecg_lavid.train --data "$OUT/ptbxl-rhythm.npz" --teacher "$OUT/ecg-muse/signature.json" \
  --out "$OUT/ecg-muse-students" --device cuda --methods ce lavid permuted \
  --seeds 17 29 43 --epochs 25 --fractions 0.1 --lavid-weight 0.4 --validation-only
```

Generate the probability-swap controls with the same questions and reference rubric. Their signature JSON files also take the trainer's `--teacher` argument.

```bash
python -m ecg_lavid.teacher_correctness --questions "$OUT/ecg-questions.json" \
  --rubric "$OUT/ecg-rubric.json" --qwen "$OUT/ecg-qwen.json" \
  --muse-audit "$OUT/ecg-muse/scoring-audit.json" --out "$OUT/ecg-teacher-controls"
python -m ecg_lavid.ecoc --out "$OUT/ecg-codebooks"
python -m ecg_lavid.train --data "$OUT/ptbxl-rhythm.npz" --codebook "$OUT/ecg-codebooks/five_way.json" \
  --out "$OUT/ecg-code-students" --device cuda --methods ce lavid \
  --seeds 17 29 43 71 101 137 --epochs 25 --fractions 0.1 --lavid-weight 0.4 --validation-only
```

The ECG codebook generator also emits a binary simplex code. Add `--shuffle-codebook-rows` with the five-way codebook to fix a class-independent assignment of the same target frequencies to records.

## 2. A–I loss controls

Let `z` be the `K` class logits, `y` the class label, `e` the one-hot label, `v` a separate `K`-output linear head, and `s` a separate head with 50 blocks of five outputs. Let `T[y]` be the fixed five-way target for class `y`, and `mean(T)` its equal-class average. `MSE` averages over all coordinates of its input. Every row predicts with `argmax(z)`.

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

The historical weighted settings use β = 50 on Flowers and β = 40 on CUB, including arms D and F. G, H and I use the same target construction. Each column of symbols is generated deterministically from integer class IDs and then softened around the uniform distribution. The softening matches one aggregate class-centered energy from the published Qwen probabilities. Thus, the codes carry class identity and a Qwen-derived global scale. The separate `ecoc-hard` ResNet-18 arm uses the fixed one-hot five-way code directly. G★ scales G to match H's auxiliary gradient norm on the shared input layer of the first training batch, then holds that scale fixed. [Squentropy](https://proceedings.mlr.press/v202/hui23a.html)

Set `FLOWERS` to the Oxford root with `jpg/`, `imagelabels.mat`, and `setid.mat`. Set `CUB` to the CUB-200-2011 root with `images/`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, and `classes.txt`. The author code uses 2,040 training and 6,149 held-out Flowers images, combining the official train and validation partitions for training. For CUB it uses the official 5,994/5,794 split. The LaViD paper appendix describes a Flowers split of 6,552/1,637. The fixed final epoch is the main comparison. Selection by lowest classification cross-entropy on the same evaluation split is recorded separately for comparison with the author code. [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) · [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) · [Author data split code](https://github.com/lliangthomas/lavid/blob/fa689286d168eb7daa9899239461cfffc5517de1/utils.py#L44-L97)

```bash
FLOWERS=data/flowers102
CUB=data/CUB_200_2011
dataset=flowers             # use cub200 with images="$CUB" for CUB
images="$FLOWERS"
QWEN="$AUTHOR/questions/$dataset/logits/qwen2.5-7b_chat_50-5.json"
CODE="$OUT/$dataset-code.json"
python -m ecg_lavid.visual_benchmark_data --dataset "$dataset" --author-source "$AUTHOR" \
  --signature "$QWEN" --out "$CODE"

arms=(ce raw-onehot prob-onehot ce-same-onehot squentropy ce-independent-onehot ecoc-soft-ce ecoc ecoc-constant)
python -m ecg_lavid.train_visual_lavid --dataset "$dataset" --images "$images" --author-source "$AUTHOR" \
  --out "$OUT/loss-resnet18" --seed 17 --arms "${arms[@]}" ecoc-soft-ce-gradmatch
python -m ecg_lavid.train_visual_loss_matrix --dataset "$dataset" --images "$images" --author-source "$AUTHOR" \
  --architecture mobilenetv2 --code "$CODE" --out "$OUT/loss-mobilenetv2" --seed 17 --arms "${arms[@]}"
```

For ViT-S/16, use `train_visual_loss_matrix --architecture vit_small_patch16 --weights <model.safetensors>` with the same cohort and code. The pinned timm weights have SHA-256 `79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa`. The CNNs use 240 epochs and batch size 16. ViT uses 500 epochs, physical batch size 64, and effective batch size 512 with cosine scheduling. For G★ on MobileNetV2 or ViT, first run with `--calibrate-gradmatch --gradient-calibration <calibration.json>` in place of `--arms`, then run separately with `--arms ecoc-soft-ce-gradmatch --gradient-calibration <calibration.json>`. [Pinned ViT-S/16 weights](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/7e2c55630205e1266030f18370f4c6ed1a514b52/model.safetensors)

The visual intervention evaluator creates a common image and donor manifest from official segmentation images. Its nine views include grayscale, masked foreground or background, background replacement, and matched controls. First generate the manifest with `--manifest-only`. Then pass the same manifest, segmentation directory, and each arm's final checkpoint. Choose `--architecture mobilenetv2` or `vit_small_patch16` for those checkpoints.

```bash
python -m ecg_lavid.visual_interventions --dataset flowers --images "$FLOWERS" \
  --masks data/flowers-segmentations --manifest "$OUT/flowers-mask-cohort.json" --manifest-only
python -m ecg_lavid.visual_interventions --dataset flowers --images "$FLOWERS" \
  --masks data/flowers-segmentations --manifest "$OUT/flowers-mask-cohort.json" \
  --author-source "$AUTHOR" --checkpoint "$OUT/loss-resnet18/flowers/seed-17/ce/final.pt" \
  --out "$OUT/flowers-ce-interventions.npz"
```

### PTB-XL loss controls

The analogous waveform experiment uses 12 five-way blocks, a fixed label-only code, and β = 0.4. The CNN and BiLSTM loss-control trainer implements A–H and G★. The missing-waveform evaluator implements paired tests for A–G and G★. The original H runs can be evaluated separately by `ecg_lavid.ptb_crossmodal_eval`.

```bash
python -m ecg_lavid.train_ptb_crossmodal --data "$OUT/ptbxl-rhythm.npz" \
  --codebook "$OUT/ecg-codebooks/five_way.json" --out "$OUT/ptb-losses" \
  --architectures cnn bilstm --fractions 1.0 --seeds 17 --device cuda \
  --arms A raw-onehot prob-onehot ce-same-onehot squentropy ce-independent-onehot ecoc-soft-ce H ecoc-soft-ce-gradmatch
python -m ecg_lavid.evaluate_ptb_loss_controls --data "$OUT/ptbxl-rhythm.npz" \
  --baselines "$OUT/ptb-losses" --runs "$OUT/ptb-losses" --out "$OUT/ptb-missing" \
  --fold 10 --missing mask --device cuda
```

## 3. Visual distillation benchmark

The hybrid trains a random-initialized vision student with the author's two-view MLKD objective and logit standardization, plus a separate weighted MSE on the weak-view student features. The fixed supervised ResNet-50 supplies weak- and strong-view logits to MLKD. The auxiliary MSE uses either the author's published Qwen2.5-7B signatures or the historical soft five-way code. The separate code-only arm uses single-view classification cross-entropy plus the five-way code MSE without a ResNet-50 teacher. All conditions use the classification head at inference. The same seed, augmentation schedule, and teacher checkpoint are used across paired hybrid conditions. [LaViD paper](https://proceedings.mlr.press/v306/liang26ac.html)

### Flowers-102 ResNet-18 reproduction

Generate the Flowers code with `visual_benchmark_data` as in section 2 and set `QWEN` and `CODE` to the Flowers files. Train the ResNet-50 with seed 17. Its checkpoint selected by held-out classification loss is the teacher for every student seed. The Flowers hybrid entry point runs 240 epochs with batch size 16, SGD with learning rate 0.01, momentum 0.9, weight decay 0.0005, and factor-0.1 learning rate drops at epochs 150, 180, and 210.

```bash
QWEN="$AUTHOR/questions/flowers/logits/qwen2.5-7b_chat_50-5.json"
CODE="$OUT/flowers-code.json"
python -m ecg_lavid.train_visual_lavid_teacher --dataset flowers --images "$FLOWERS" \
  --author-source "$AUTHOR" --out "$OUT/visual-teachers" --seed 17 --device cuda
TEACHER="$OUT/visual-teachers/flowers/seed-17/author-best.pt"
for seed in 17 29 43; do
  python -m ecg_lavid.train_visual_lavid_ls --images "$FLOWERS" --author-source "$AUTHOR" \
    --out "$OUT/flowers-hybrids" --teacher-checkpoint "$TEACHER" --target-kind qwen --target-file "$QWEN" \
    --seed "$seed" --device cuda
done
for seed in 17 29 43; do
  python -m ecg_lavid.train_visual_lavid_ls --images "$FLOWERS" --author-source "$AUTHOR" \
    --out "$OUT/flowers-hybrids" --teacher-checkpoint "$TEACHER" --target-kind ecoc --target-file "$CODE" \
    --seed "$seed" --device cuda
  python -m ecg_lavid.train_visual_benchmark --dataset flowers --architecture resnet18 --method ecoc-only \
    --images "$FLOWERS" --author-source "$AUTHOR" --target "$CODE" --out "$OUT/visual-grid" \
    --seed "$seed" --device cuda
done
```

To reuse the Qwen run's BF16 vision-teacher logits in its paired code run, pass the same `--teacher-logit-cache <directory>` to both hybrid calls. The Qwen call records and the code call replays. For original-size RGB caching, run `python scripts/build_flowers_rgb_cache.py --images "$FLOWERS" --out "$OUT/flowers-rgb-cache"`, then pass that folder as `--training-rgb-cache` to the vision teacher and both hybrid calls.

### Extend to other students and datasets

`ecg_lavid.train_visual_benchmark` supports the author ResNet-18, MobileNetV2, and ShuffleNetV2 x1.0. For a paired hybrid, run `lavid-ls-qwen` to completion before `lavid-ls-ecoc`, using the same output root for that pair. The two methods use `--teacher-checkpoint` and differ in `--target`. If caching, add `--record-teacher-logits` only to the Qwen command and give both runs the same `--teacher-logit-cache` path. `ecoc-only` needs the generated five-way target and no visual checkpoint.

| CLI dataset | Required layout and training/test sizes |
| --- | --- |
| `flowers` | Oxford image and MAT files above. 2,040 / 6,149. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| `cub200` | Official CUB files above. 5,994 / 5,794. [Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/) |
| `fgvc_aircraft` | TorchVision layout with `fgvc-aircraft-2013b/data/` and `images/`. Use variant-level trainval/test. 6,667 / 3,333. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) |
| `oxfordpets` | TorchVision layout under `oxford-iiit-pet/` with images and annotations. Use trainval/test. 3,680 / 3,669. [Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) |
| `stanfordcars` | TorchVision layout under `stanford_cars/` with both splits, the devkit and test labels. 8,144 / 8,041. [TorchVision recovery information](https://github.com/pytorch/vision/issues/7545) |
| `caltech101` | Root with `101_ObjectCategories/`. Use the author's split seed 27 after sorting files within each class. Both face labels are retained and share a language signature. 4,310 / 4,367. [Dataset](https://data.caltech.edu/records/mzrjq-6wc02) |

For Aircraft, Pets, Cars, and Caltech, generate the split once with `scripts/prepare_benchmark_cohort.py` and pass its `cohort.csv` to code generation, teacher training, and student training. These folders also contain the corresponding `cohort.json` sidecar. For the first five datasets, `--signature` points to `$AUTHOR/questions/<dataset>/logits/qwen2.5-7b_chat_50-5.json`. The author's Caltech release is labeled 72B. The separate `scripts/reconstruct_caltech7b.py` entry point runs the author's scorer using pinned Qwen2.5-7B weights and checks a Flowers subset for calibration before producing a Caltech 7B candidate. A candidate's provenance and calibration scores should accompany comparisons with the paper. [Published Caltech scores](https://github.com/lliangthomas/lavid/blob/fa689286d168eb7daa9899239461cfffc5517de1/questions/caltech101/logits/qwen2.5-72B_chat_50-5.json)

For example, the following commands run the paired MobileNetV2 arms on CUB.

```bash
QWEN="$AUTHOR/questions/cub200/logits/qwen2.5-7b_chat_50-5.json"
CODE="$OUT/cub200-code.json"
python -m ecg_lavid.visual_benchmark_data --dataset cub200 --author-source "$AUTHOR" \
  --signature "$QWEN" --out "$CODE"
python -m ecg_lavid.train_visual_lavid_teacher --dataset cub200 --images "$CUB" \
  --author-source "$AUTHOR" --out "$OUT/visual-teachers" --seed 17 --device cuda
TEACHER="$OUT/visual-teachers/cub200/seed-17/author-best.pt"
python -m ecg_lavid.train_visual_benchmark --dataset cub200 --architecture mobilenetv2 --method lavid-ls-qwen \
  --images "$CUB" --author-source "$AUTHOR" --target "$QWEN" --teacher-checkpoint "$TEACHER" \
  --out "$OUT/visual-grid" --seed 17 --device cuda
python -m ecg_lavid.train_visual_benchmark --dataset cub200 --architecture mobilenetv2 --method lavid-ls-ecoc \
  --images "$CUB" --author-source "$AUTHOR" --target "$CODE" --teacher-checkpoint "$TEACHER" \
  --out "$OUT/visual-grid" --seed 17 --device cuda
```
