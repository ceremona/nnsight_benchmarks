# NNsight Benchmarks

Interpretability experiments with [NNsight](https://github.com/NDIF/nnsight): ablate a layer, measure what changes. Two notebooks, one question — *where does a model's behavior actually live?*

| Notebook | Model | Question |
|---|---|---|
| [`nnsight_benchmarks.ipynb`](nnsight_benchmarks(4).ipynb) | GPT-2 (124M) | What happens to a factual prediction when you zero out a single layer? |
| [`nnsight_topology_refusal.ipynb`](nnsight_topology_refusal.ipynb) | Qwen2.5-1.5B-Instruct | Does refusal have a *shape*? Persistent homology on activation point clouds. |

## The thread

Both notebooks do the same thing at different resolutions:

1. **Perturb** — zero out a layer's output mid-forward-pass via NNsight tracing.
2. **Measure** — quantify the downstream disruption (logit lens, KL divergence, Betti curves, bottleneck distance).
3. **Localize** — find which layers are load-bearing.

The GPT-2 notebook is the didactic version: one prompt, one ablation, clean visuals. The topology notebook scales the idea up: instead of tracking a single prediction, it asks whether the *geometry of an entire activation cloud* (harmful vs. harmless prompts) carries detectable structure — using Vietoris–Rips persistent homology rather than directional probes.

The betti-curve machinery here is the activation-space analog of graph-based Betti analysis on coupling matrices: build a similarity structure, threshold it, count holes. If refusal is a low-dimensional direction (cf. Arditi et al., 2024), it should leave a topological signature.

## Notebooks

### 1. NNsight tests — Visualized (`nnsight_benchmarks.ipynb`)

Layer ablation on `"The Eiffel Tower is in the city of"` (baseline prediction: `Paris`).

- **Logit lens table** — per-layer next-token predictions, baseline vs. ablated
- **Activation norm heatmaps** — L2 norms across layers × tokens, with a difference panel showing disruption propagation
- **Probability shift** — top-12 next-token probabilities before/after ablation
- **Layer importance sweep** — ablate each layer one at a time, plot KL divergence from baseline

Finding: ablating layer 5 flips `Paris` → `and`; layers ~3–6 and 8 carry most of the predictive weight for this prompt.

### 2. Refusal Topology (`nnsight_topology_refusal.ipynb`)

40 harmful + 40 harmless prompts through Qwen2.5-1.5B-Instruct; last-token hidden states captured at layer 14, optionally with layer 10 ablated.

- **Persistent homology** (ripser) on PCA-reduced activation clouds, `maxdim=2`
- **Persistence diagrams** — H0/H1/H2 features, harmful vs. harmless vs. combined
- **Betti curves** — components, loops, and voids vs. distance threshold
- **Bottleneck distance** between H1 diagrams as a topology-change metric
- **Ablation experiment** — does zeroing an earlier layer collapse the refusal-relevant structure?

⚠️ Didactic exercise: small dataset, and ripser gets expensive fast on free Colab hardware.

## Setup

Both notebooks run on a **free Google Colab T4**.
