## ⚡ SLM Benchmark

**A focused, transparent benchmark for Small Language Models (≈1M–3B parameters).**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Website](https://img.shields.io/badge/leaderboard-live-green.svg)](https://2796gaurav.github.io/slm-benchmark)

### 🎯 What this project is

- **Small‑model first**: a leaderboard built specifically for SLMs (up to ~3B parameters), not giant frontier models.
- **Unbiased by hardware**: the main ranking only uses accuracy‑style scores; latency, energy, and CO₂ are reported but
  never used to reorder models.
- **Transparent and inspectable**: every run produces JSON artifacts; the public registry and website data are
  version‑controlled.

If you just want to browse the results, open the live site: `https://2796gaurav.github.io/slm-benchmark`.

---

## 🚀 TL;DR for visitors

- **Leaderboard**: each row shows:
  - Model name, family, and Hugging Face link.
  - Aggregate score and per‑pillar scores (Reasoning, Coding, Math, Language, Safety, Edge).
  - Informational metrics such as efficiency and estimated CO₂.
  - Badges like **SLM‑Optimized**, **Eco‑Efficient**, **Safety‑Preferred** (non‑normative hints, not labels).
- **Methodology**: see `website/methodology.html` or the “Benchmark Methodology” section below.
- **How to submit a model**: see the next section, or the `website/submit.html` page on the live site.

You do *not* need to run anything locally to read the leaderboard.

---

## 📝 How to submit your model

There are two ways to get a model evaluated:

- **For most users (recommended)** — open a GitHub Issue with a YAML snippet.
- **For maintainers / power users** — run the benchmark yourself and update the registry.

### 1. Describe your model in YAML

Use this minimal schema:

```yaml
model:
  name: "SmolLM2-1.7B"
  family: "SmolLM"
  hf_repo: "HuggingFaceTB/SmolLM2-1.7B"
  parameters: "1.7B"           # must be ≤ 3B
  architecture: "llama"
  context_length: 8192
  license: "Apache-2.0"

  # Evaluation pillars you want to be considered in the aggregate score
  categories:
    - "reasoning"
    - "coding"
    - "math"
    - "language"
    - "safety"

  # Optional: quantizations that were evaluated
  quantizations:
    - name: "FP16"
      format: "safetensors"

  # Submission metadata
  submitted_by: "github_username"
  contact: "email@example.com"
```

**Hard requirements**

- Model size ≤ 3B parameters.
- Model is public on Hugging Face.
- License explicitly allows benchmarking and publishing results.

For concrete examples, see `models/submissions/template.yml` and `models/submissions/tiny_test.yaml`.

### 2. Open a “Model Submission” issue

1. Go to the GitHub repository.
2. Open a new issue using the **“Model Submission”** template.
3. Paste your YAML into the description.

The benchmark pipeline will:

- Validate basic metadata (size, license, categories).
- Schedule an evaluation run.
- Update the internal registry and `website/assets/data/leaderboard.json`.

Updates usually appear on the public leaderboard once maintainers or community runners have processed the run.

### 3. (Optional) Run the benchmark yourself

If you prefer to run everything locally and contribute results:

```bash
git clone https://github.com/2796gaurav/slm-benchmark.git
cd slm-benchmark

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

python benchmarks/evaluation/run_benchmark.py \
  --submission-file models/submissions/tiny_test.yaml \
  --output-dir results/raw/ \
  --limit 5 \
  --batch-size 1
```

Then use the scripts under `scripts/` (`generate_report.py`, `update_registry.py`, `update_website.py`) to fold the run
into `models/registry.json` and refresh `website/assets/data/leaderboard.json`. See `DEVELOPER_GUIDE.md` for exact
commands.

---

## 📊 Benchmark methodology (high level)

### Evaluation pillars

- **Reasoning** — tasks like MMLU, ARC‑Challenge, HellaSwag, TruthfulQA.
- **Coding** — tasks like HumanEval, MBPP (with safe‑execution flags).
- **Math** — tasks like GSM8K / math QA.
- **Language** — tasks like BoolQ, PIQA, WinoGrande.
- **Safety** — toxicity, bias, truthfulness, and fairness‑oriented probes.
- **Long context** *(informational)* — needle‑in‑haystack and multi‑document QA (not part of the aggregate score).
- **Edge / efficiency** *(informational)* — latency, memory, energy, and CO₂ estimates.

All pillars are implemented via `benchmarks/evaluation/*.py` on top of `lm-evaluation-harness` (`HFLM` backend) plus
additional safety and bias checks.

### Aggregate score (used for ranking)

The aggregate ranking score only uses accuracy‑style pillars:

- Reasoning — 35%
- Coding — 20%
- Math — 15%
- Language — 20%
- Safety — 20%

Each pillar is normalized to \[0, 100\], then combined:

\\[
Score_{final} = 0.35 \cdot R + 0.20 \cdot C + 0.15 \cdot M + 0.20 \cdot L + 0.20 \cdot S
\\]

where \(R, C, M, L, S\) are average scores per pillar.

**Hardware‑dependent metrics (latency, throughput, energy, CO₂) are never included in this formula.** They are still
recorded and shown on the website so users can decide their own trade‑offs, but they do not affect rank.

### Unbiasedness and fairness philosophy

- **Hardware neutrality**: rankings do not depend on which GPU/CPU ran the benchmark; only accuracy‑like scores matter.
- **Task diversity**: we mix reasoning, coding, math, language, and safety so that no single style of model dominates.
- **Explicit safety reporting**: safety and fairness signals are surfaced as a first‑class pillar rather than hidden
  footnotes.
- **Transparent artifacts**: every run keeps full JSON artifacts, so anyone can recompute alternative scores or audit
  the results.

For implementation details, see `benchmarks/evaluation/run_benchmark.py` and `DEVELOPER_GUIDE.md`.

---

## 🧱 Repository layout (for developers)

```text
slm-benchmark/
  benchmarks/
    evaluation/        # Benchmark runner and evaluation modules
    configs/           # High-level configs
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
    ci.yml             # Test workflow
```

If you want to extend or integrate with the benchmark, start with `DEVELOPER_GUIDE.md`.

---

## 🧩 Docs & support

- **How to contribute**: `CONTRIBUTING.md`
- **Developer internals**: `DEVELOPER_GUIDE.md`
- **Troubleshooting runs**: `TROUBLESHOOTING.md`
- **Support & contact**: `SUPPORT.md` and the `website/support.html` page

Maintainer: **@2796gaurav** (GitHub). Please prefer public Issues / Discussions so others can learn from the answers.

---

## 📜 License & acknowledgements

- License: **Apache 2.0** (see `LICENSE`).
- Built on:
  - **EleutherAI** — `lm-evaluation-harness`
  - **Hugging Face** — Transformers and model hosting
  - **CodeCarbon** — optional emission and energy tracking

<div align="center">

**⚡ Built for the SLM community.**

[![Star on GitHub](https://img.shields.io/github/stars/2796gaurav/slm-benchmark?style=social)](https://github.com/2796gaurav/slm-benchmark)

</div>
