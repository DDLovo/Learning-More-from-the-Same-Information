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
