# Learning More from the Same Information

Can a classifier learn more from each labeled example if we ask it to predict the same label in several different ways?

We began with ECG rhythm classification. A language model can answer questions about a rhythm's expected heart rate, atrial activity, and beat regularity. A waveform classifier can learn to predict those answers alongside the rhythm label. This raises a question about where any gain comes from. How much depends on the medical meaning of the answers? How much can be recovered by giving each class a consistent set of auxiliary targets?

## How a class becomes an auxiliary task

For each position in a code, we assign each class one of five symbols. Classes that share a symbol at one position can have different symbols at another. The assignments stay fixed throughout training, so each position asks the student to distinguish a different grouping of the same classes. A classification head predicts the original label, and a second head predicts its code from the same representation. At inference, the student uses the classification head alone.

The symbol assignments need only class IDs. The ECG experiments use hard targets. For the visual comparisons reported below, the codes use a common softening factor calibrated to the overall variation of Qwen's answers within each dataset. This matches the scale of the supervision while removing Qwen's answer to each individual question.

## Experiments

| Experiment | Question |
| --- | --- |
| [ECG rhythm classification](experiments/ecg-distillation/) | On PTB-XL, how do Qwen and Muse answers compare with answers reassigned across rhythms and with fixed class codes? |
| [A–I loss controls](experiments/loss-controls/) | Does the gain depend on the code, the extra head, or the loss? One control gives every class the same target. Image interventions measure how accuracy changes when foreground or background cues are disrupted. |
| [Fine-grained visual distillation](experiments/visual-distillation/) | With the same ResNet-50 teacher, how much does a student lose when Qwen's answers are replaced by a five-way code? The benchmark also evaluates code supervision on its own. |

On Flowers and CUB, the [paired foreground analysis](experiments/loss-controls/gradcam/) asks a complementary question. After a background is removed or replaced, how does the spatial evidence for the correct species change? It compares A and H on the same images and relates their Grad-CAM maps to the released foreground masks.

## What we observed on Flowers-102

Each row uses a ResNet-18 student. Values are mean top-1 accuracy over the same three seeds at epoch 240, using the public LaViD code's split of 2,040 training and 6,149 evaluation images. [LaViD paper](https://proceedings.mlr.press/v306/liang26ac.html) · [Upstream implementation](https://github.com/lliangthomas/lavid/tree/fa689286d168eb7daa9899239461cfffc5517de1)

| Training protocol | Cross-entropy alone | With Qwen answers | With five-way codes |
| --- | ---: | ---: | ---: |
| Single-view training | 76.24% | 80.51% | 80.48% |
| Two-view distillation with a ResNet-50 teacher | — | 85.15% | 84.62% |

In single-view training, the five-way code improved accuracy by 4.24 percentage points over cross-entropy and came within 0.03 points of Qwen. With the visual teacher held fixed, replacing Qwen cost 0.53 points on average. Whether the auxiliary task helps the student use additional discriminative cues remains an open question. The loss controls, image interventions, and comparisons across datasets and architectures test different parts of that explanation.
