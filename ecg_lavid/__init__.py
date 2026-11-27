"""Cross-modal distillation for PTB-XL rhythm classification."""

CLASSES = {
    "SR": "sinus rhythm",
    "AFIB": "atrial fibrillation",
    "STACH": "sinus tachycardia",
    "SBRAD": "sinus bradycardia",
}
LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
SAMPLE_RATE = 100
N_SAMPLES = 1_000
