"""Compatibility entry point for the visual distillation experiment."""

from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    script = Path(__file__).resolve()
    run_path(str(script.parents[1] / "experiments" / "visual-distillation" / script.name), run_name="__main__")
