## ⚡ SLM Benchmark

**CPU-first, transparent benchmarking platform for Small Language Models (≈1M–3B parameters).**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Website](https://img.shields.io/badge/leaderboard-live-green.svg)](https://2796gaurav.github.io/slm-benchmark)

### 🎯 Mission

Provide a **practical, CPU-friendly, and unbiased** evaluation platform for Small Language Models, so developers can
pick the right model for edge devices and resource‑constrained environments.

---

### ✨ What You Get

- **Hardware-agnostic ranking**: Aggregate scores ignore raw latency/throughput so results remain comparable across
  machines (including GitHub Actions runners).
- **SLM-focused tasks**: Reasoning, coding, math, language understanding, safety, and long‑context checks tuned for
  small models.
- **Transparent artifacts**: JSON outputs, registry, and website data are all version‑controlled.
- **Zero build frontend**: Static HTML/CSS/JS leaderboard that reads a single `leaderboard.json` file.

---

## 🚀 Quick Start

### View the Leaderboard

- Open the live site: `https://2796gaurav.github.io/slm-benchmark`
- Each row shows:
  - Model name, family, Hugging Face link.
  - Aggregate score and per‑pillar scores.
  - Efficiency / CO₂ metrics (informational only).
  - Badges such as **SLM‑Optimized**, **Eco‑Efficient**, **Safety‑Preferred**.

### Run a Tiny CPU Smoke Test Locally

```bash
git clone https://github.com/2796gaurav/slm-benchmark.git
cd slm-benchmark

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

# Force CPU
export CUDA_VISIBLE_DEVICES=""

# Run a tiny benchmark on a small public model
python benchmarks/evaluation/run_benchmark.py \
  --submission-file models/submissions/tiny_test.yaml \
  --output-dir results/raw/ \
  --limit 5 \
  --batch-size 1
```

Artifacts will appear under `results/raw/TinyStories-1M-Verify/<timestamp>/`.

---

## 📝 Submission Format (YAML)

Create a file in `models/submissions/`:

```yaml
model:
  name: "SmolLM2-1.7B"
  family: "SmolLM"
  version: "2.0"

  # Hugging Face details
  hf_repo: "HuggingFaceTB/SmolLM2-1.7B"
  hf_revision: "main"  # or a specific commit

  # Model specifications
  parameters: "1.7B"         # must be ≤ 3B
  architecture: "llama"
  context_length: 8192
  license: "Apache-2.0"

  # Quantizations to record (metadata only for now)
  quantizations:
    - name: "FP16"
      format: "safetensors"

  # Evaluation pillars (select all that apply)
  categories:
    - "reasoning"
    - "coding"
    - "math"
    - "language"
    - "safety"

  # Submission metadata
  submitted_by: "github_username"
  submitted_date: "2026-01-13"
  contact: "email@example.com"
```

**Requirements**

- Model size ≤ 3B parameters.
- Model is public on Hugging Face.
- License explicitly allows benchmarking and publishing results.

For a minimal example, see `models/submissions/template.yml` and `models/submissions/tiny_test.yaml`.

---

## 📊 Benchmark Methodology (2026)

### Deterministic, CPU-Friendly Runs

- Fixed random seed: `42`.
- Evaluation backend: `lm-evaluation-harness` via `HFLM` (Transformers).
- Works on **CPU by default**; if a GPU is present, it may be used for speed but **scores are designed to be
  hardware‑agnostic**.

### Evaluation Pillars

- **Reasoning** — tasks like MMLU, ARC‑Challenge, HellaSwag, TruthfulQA.
- **Coding** — tasks like HumanEval, MBPP (run with safe execution flags).
- **Math** — tasks like GSM8K / Math QA.
- **Language** — tasks like BoolQ, PIQA, WinoGrande.
- **Safety** — lightweight toxicity, bias, and truthfulness probes (with optional Detoxify integration).
- **Long context (informational)** — Needle‑in‑haystack and multi‑doc QA via `LongContextEvaluator`.

### Aggregate Score (Used for Ranking)

Hardware‑dependent metrics (latency, throughput, energy, CO₂) are **not** used in the aggregate score.

We compute:

- Reasoning — 35%
- Coding — 20%
- Math — 15%
- Language — 20%
- Safety — 20%

Each pillar is normalized to \[0, 100\], then combined:

\\[
Score_{final} = 0.35 \cdot R + 0.20 \cdot C + 0.15 \cdot M + 0.20 \cdot L + 0.20 \cdot S
\\]

Where \(R, C, M, L, S\) are the average scores per pillar.

### Hardware-Dependent Metrics (Informational Only)

When enabled, we also report:

- **Latency / Throughput** (EdgeBenchmark).
- **Memory usage** (CPU or GPU if available).
- **Energy & CO₂** (CodeCarbon via `CarbonTrackerWrapper`).
- **Heuristic fine‑tuning friendliness** (no real fine‑tuning runs).

These appear in the JSON, registry, and website, but are **not** part of the aggregate ranking.

---

## 🧱 Repository Layout

```text
slm-benchmark/
  benchmarks/
    evaluation/        # Benchmark runner and evaluation modules
    configs/           # High-level configs (currently informational)
    validation/        # Auto-detection tools for HF models
  models/
    registry.json      # Source of truth for leaderboard
    submissions/       # Model submission YAMLs
  results/
    raw/               # Per-run JSON artifacts
    processed/         # Aggregated reports
    archives/          # Historical snapshots
  scripts/
    update_registry.py # Registry updater
    update_website.py  # Syncs registry -> website JSON
    generate_report.py # Aggregates raw results
  website/
    index.html, model.html, methodology.html, submit.html, support.html
    assets/css, assets/js, assets/data/leaderboard.json
  .github/workflows/
    ci.yml             # CPU-only test workflow
```

See `DEVELOPER_GUIDE.md` for a deeper architectural walkthrough.

---

## 🤖 GitHub Actions & CPU CI

- `.github/workflows/ci.yml` runs:
  - `pip install -r requirements.txt`
  - `pytest`
- Defaults:
  - `CUDA_VISIBLE_DEVICES` is cleared so runs do not assume a GPU.
  - Tests are designed to be fast and network‑light (no heavy HF downloads).

You can mirror this pattern in your own repositories when using SLM Benchmark as a reference.

---

## 🔖 Badges & Model Metadata

The website dynamically assigns **non‑normative badges** based on JSON metadata:

- **SLM‑Optimized** — parameters ≤ 500M.
- **Eco‑Efficient** — high efficiency score (good accuracy / kWh where energy data is available).
- **Safety‑Preferred** — safety pillar score ≥ 90.

Badges are meant as guidance, not absolute labels; for research‑grade analysis always inspect the full JSON metrics.

---

## 🧩 Docs & Support

- **Contribution guidelines**: `CONTRIBUTING.md`
- **Developer guide**: `DEVELOPER_GUIDE.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`
- **Support**: `SUPPORT.md` (and the `website/support.html` page)

Maintainer: **@2796gaurav** (GitHub). Please prefer GitHub Issues / Discussions for questions.

---

## 📜 License & Acknowledgements

- License: **Apache 2.0** (see `LICENSE`).
- Built on:
  - **EleutherAI** — `lm-evaluation-harness`
  - **Hugging Face** — Transformers and model hosting
  - **CodeCarbon** — emission and energy tracking (optional)

---

<div align="center">

**⚡ Built with care for the SLM community.**

[![Star on GitHub](https://img.shields.io/github/stars/2796gaurav/slm-benchmark?style=social)](https://github.com/2796gaurav/slm-benchmark)

</div>
