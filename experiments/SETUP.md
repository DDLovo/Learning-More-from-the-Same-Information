# Experiment setup

Use Python 3.11 or newer with a matched PyTorch and TorchVision build. Full visual training and the pinned language-model scoring runs use CUDA with BF16 support. Run the shell commands from the repository root in Bash. On Windows, set `WORKERS=0` for the visual launchers.

```bash
python -m pip install -r requirements.txt
mkdir -p data runs
```

The launchers use `python` and `cuda` by default. Set `PYTHON` to the path of another Python executable or `DEVICE` to another PyTorch device. Use an output directory such as `runs/ecg` for generated inputs, training results, and checkpoints.

## LaViD source for the visual experiments

Use the following checkout for the image loss controls and visual-distillation benchmark. Turning off Git's line-ending conversion preserves the upstream byte hashes on Windows.

```bash
git -c core.autocrlf=false clone https://github.com/lliangthomas/lavid.git lavid-upstream
git -C lavid-upstream -c core.autocrlf=false checkout --detach fa689286d168eb7daa9899239461cfffc5517de1
```

The checkout supplies the author model definitions, augmentation, MLKD objective, ordered class names, and visual Qwen signatures. Pass its path as `AUTHOR_SOURCE` to the visual launchers. [LaViD paper](https://proceedings.mlr.press/v306/liang26ac.html) · [Pinned author source](https://github.com/lliangthomas/lavid/tree/fa689286d168eb7daa9899239461cfffc5517de1)
