## Developer Guide

This guide is for maintainers and contributors who want to extend or integrate with **SLM Benchmark**.

### High-Level Architecture

- `benchmarks/evaluation/run_benchmark.py` — main entrypoint, orchestrates evaluation for a submission YAML.
- `benchmarks/evaluation/*.py` — individual evaluation modules (edge metrics, safety, long context, etc.).
- `benchmarks/validation/auto_detector.py` — auto-detects model metadata from a Hugging Face repo.
- `models/registry.json` — single source of truth for the leaderboard models and scores.
- `scripts/*.py` — utility scripts for updating the registry, website data, and aggregating reports.
- `website/` — static HTML/CSS/JS GitHub Pages site that reads `assets/data/leaderboard.json`.

### Benchmark Flow (CPU-Friendly)

1. User creates a YAML under `models/submissions/` (see `template.yml`).
2. `run_benchmark.py` is invoked with:
   ```bash
   python benchmarks/evaluation/run_benchmark.py \
     --submission-file models/submissions/my-model.yaml \
     --output-dir results/raw/ \
     --limit 5
   ```
3. For each quantization entry in the YAML, the runner:
   - Loads the model via **lm-evaluation-harness HFLM** (Transformers backend).
   - Runs reasoning/coding/math/language benchmarks (via `lm_eval.evaluator.simple_evaluate`).
   - Optionally runs safety, long-context, and edge (hardware-dependent) benchmarks.
   - Writes per-quantization JSON results and a `summary.json` under `results/raw/<model>/<timestamp>/`.
4. `scripts/generate_report.py` can aggregate multiple quantizations into a single JSON.
5. `scripts/update_registry.py` merges a processed result into `models/registry.json`.
6. `scripts/update_website.py` copies `registry.json` to `website/assets/data/leaderboard.json`.

### Hardware-Agnostic Scoring

- The **aggregate ranking score** only uses:
  - Reasoning, Coding, Math, Language, Safety.
- Hardware-dependent metrics (latency, throughput, memory, energy, carbon) are:
  - Still computed (if enabled) and shown in JSON / website.
  - **Not used** in ranking or badges that imply fairness across hardware.

If you add a new metric, decide explicitly:
- Is it hardware-dependent? If yes, keep it out of the aggregate ranking.
- Does it belong in `scores` (scalar) or in a separate nested object?

### Adding a New Evaluation Category

1. Implement a new method on `SLMBenchmark` or a new module under `benchmarks/evaluation/`.
2. Return a dictionary mapping task name → score dict (ideally with `acc` key if you want it averaged).
3. Optionally expose a scalar by:
   - Updating `scripts/update_registry.py` to compute `_average_score` over the new dict.
4. Update the website:
   - Add a new column in `website/index.html` table header.
   - Extend `leaderboard.js` sorting and rendering.

### GitHub Actions (CPU CI)

- The repo includes `.github/workflows/ci.yml` which:
  - Installs dependencies.
  - Runs `pytest` on Ubuntu with `CUDA_VISIBLE_DEVICES=""`.
- Keep tests:
  - Fast (no heavy benchmarks or Hugging Face downloads).
  - Deterministic (no random network dependencies).

### Design Principles

- **Small-model first**: always consider 1M–5B models on commodity CPUs as the primary target.
- **Reproducible**: deterministic seeds, explicit configs, and JSON artifacts.
- **Transparent**: any change to scoring or methodology must be reflected in `README.md` and this guide.










