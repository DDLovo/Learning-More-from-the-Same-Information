# ECG language distillation

The task uses 10-second, 12-lead, 100 Hz recordings from PTB-XL v1.0.3 with one annotated rhythm in the four-class cohort `SR`, `AFIB`, `STACH`, and `SBRAD`. Official folds 1–8 train the waveform classifier, fold 9 selects the checkpoint by macro-F1, and fold 10 provides the held-out test. The low-label setting subsamples training patients and fits per-lead normalization on that subset. [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

The teachers receive class names and fixed questions about ventricular rate, R–R intervals, atrial activity, and P–QRS timing. The student learns the true rhythm with cross-entropy and uses a separate linear head to regress the corresponding class signature. The `permuted` arm reassigns whole signatures using a fixed class derangement. All students use their waveform classification head for evaluation.

## Prepare and run

Install the [shared dependencies](../SETUP.md), then run the [experiment launcher](run.sh) from the repository root. Place `ptbxl_database.csv`, `scp_statements.csv`, and `records100/` under `data/ptb-xl-1.0.3`. Alternatively, pass a final argument of `16` to `prepare` to download the missing metadata and cohort waveform files in parallel. The preparation stage also generates the fixed Astra question bank and rubric. Use `questions` to regenerate just those two files.

```bash
bash experiments/ecg-distillation/run.sh prepare runs/ecg data/ptb-xl-1.0.3
bash experiments/ecg-distillation/run.sh score-qwen runs/ecg
bash experiments/ecg-distillation/run.sh train-qwen runs/ecg
```

Qwen scores five single-letter continuations at the assistant's first token and normalizes their logits with a temperature of one. The launcher uses Qwen3-4B-Instruct-2507 at revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The earlier scoring control used Qwen2.5-1.5B-Instruct at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. To run that control, pass the model and revision to `ecg_lavid.teacher`. The `ecg_lavid.rhythm_question_bank --bank manual` entry point materializes the original manually prepared questions. For a new bank, run `ecg_lavid.teacher` without `--questions` or use `ecg_lavid.generate_openai` with a supported model and `OPENAI_API_KEY`. [Pinned Qwen3 weights](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) · [Pinned Qwen2.5 weights](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

### Muse Spark Open

Provide the credential through `OPENROUTER_API_KEY` before scoring. This step uses the `meta/muse-spark-open-20261015` model on `nebius/fp4` and requires support for top-20 token log-probabilities. The audit locates the first visible token after any returned internal tokens. For missing bare options outside a verified top-20 list, it reports bounds on the missing probability mass. The signature estimates each missing option at half the cutoff probability and then renormalizes over A–E. [OpenRouter model page](https://openrouter.ai/meta/muse-spark-open)

```bash
bash experiments/ecg-distillation/run.sh score-muse runs/ecg
bash experiments/ecg-distillation/run.sh train-muse runs/ecg
```

Both student stages use seeds 17, 29, and 43 for 25 epochs at 10% of training patients with auxiliary weight 0.4. They run the CE, signature, and permuted-signature arms and score on fold 9. The run configuration records the target fingerprint, patient subset, seed, schedule, and class permutation. Use the trainer directly to evaluate on fold 10 or change the training budget.

### Muse text reports and native token probabilities

The launcher generates the frozen discriminative bank of twelve questions and uses it for both scoring modes with the same dated model and provider. Each native response supplies the first visible token distribution over A–E. Each text response is a JSON object with a declared answer and five self-reported probabilities. The scorer preserves the first valid response for each item and writes both soft signatures, both one-hot signatures, a request manifest, and metrics against the frozen rubric. Requests contain the public class name and question text. [Text probability scorer](../../ecg_lavid/muse_text_probability.py)

```bash
bash experiments/ecg-distillation/run.sh score-muse-text runs/ecg
bash experiments/ecg-distillation/run.sh train-muse-text runs/ecg
```

The student stage uses the same CNN recipe at 10% of training patients for seeds 17, 29, 43, 71, 101, and 137. It shares the hard-target runs when the two one-hot signatures match exactly. All checkpoints are selected and scored on fold 9 for this comparison. The scorer's `--native-only` switch rebuilds the two native targets from a completed scoring directory. The optional `--bundle` switch collects the separate teacher-only condition in which one request contains all 48 items.

### Teacher correctness and code controls

After obtaining both signatures, `controls` writes the reference-answer and matched-argmax probability-swap controls under `runs/ecg/ecg-teacher-controls/`. Each signature is accepted by the trainer's `--teacher` argument. The `codes` stage generates the five-way and binary simplex codebooks and runs a CE/five-way comparison on six paired seeds. The trainer also accepts the binary simplex file with `--codebook`. With the five-way codebook, `--shuffle-codebook-rows` fixes a class-independent assignment of the same target frequencies to records.

```bash
bash experiments/ecg-distillation/run.sh controls runs/ecg
bash experiments/ecg-distillation/run.sh codes runs/ecg
```

The following example trains on the Qwen target and reports the held-out fold using a separate output directory. The same trainer accepts the Muse signature at `runs/ecg/ecg-muse/signature.json` or a generated probability-swap signature.

```bash
python -m ecg_lavid.train --data runs/ecg/ptbxl-rhythm.npz --teacher runs/ecg/ecg-qwen.json \
  --out runs/ecg-qwen-test --device cuda --methods ce lavid permuted \
  --seeds 17 29 43 --epochs 25 --fractions 0.1 --lavid-weight 0.4
```

## Record-level R–R interval targets

The fixed two-question bank covers the Uni-G mean sinus R–R interval (`RR_Mean_Global`) and the standard deviation of normal R–R intervals (`RR_StdDev_Global`). The trainer joins released PTB-XL+ v1.0.1 measurements to the prepared ECG IDs and fits the five answer bins on each seed's training patients. It compares rhythm CE with CE plus matched targets and CE plus whole-row targets from another patient in the same rhythm and validity group. The latter two arms share the binning and valid-record mask. The student receives the waveform, and the auxiliary labels come from the released measurements. [PTB-XL+ v1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/) · [Uni-G field definitions](https://physionet.org/files/ptb-xl-plus/1.0.1/features/feature_description.csv)

```bash
mkdir -p data/ptb-xl-plus
curl --fail --location https://physionet.org/files/ptb-xl-plus/1.0.1/features/unig_features.csv \
  --output data/ptb-xl-plus/unig_features.csv
bash experiments/ecg-distillation/run_rr.sh fit runs/ecg/ptbxl-rhythm.npz \
  data/ptb-xl-plus/unig_features.csv runs/rr
bash experiments/ecg-distillation/run_rr.sh replay runs/ecg/ptbxl-rhythm.npz runs/rr
```

The launcher runs the completed 10% and full-training-patient comparisons for seeds 17, 29, and 43. It selects each checkpoint using fold 9 rhythm macro-F1 and evaluates on fold 10. The trainer reports a frozen per-rhythm ridge readout, with standardization and ridge fitting confined to training patients. The replay uses the recorded batch size and precision and checks the reproduced auxiliary probabilities against the saved predictions. It saves the two raw five-way output vectors and their fixed ordinal projection. Use the training device and runtime build for exact replay. [RR launcher](run_rr.sh)

The released global ECGDeli peaks provide a second reference on the same PTB-XL test records. LUDB v1.0.1 supplies an external check using the manually annotated lead-II QRS peaks. The preparation stages fetch annotations and waveforms from their pinned releases and keep generated files under the supplied directories. [PTB-XL+ ECGDeli release](https://physionet.org/files/ptb-xl-plus/1.0.1/fiducial_points/ecgdeli/) · [LUDB v1.0.1](https://physionet.org/content/ludb/1.0.1/)

```bash
bash experiments/ecg-distillation/run_rr.sh ecgdeli data/ptb-xl-plus \
  data/ptb-xl-plus/unig_features.csv runs/rr
bash experiments/ecg-distillation/run_rr.sh ludb data/ludb-1.0.1 runs/rr
```
