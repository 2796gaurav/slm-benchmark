## Troubleshooting SLM Benchmark

This guide lists common problems and how to fix them when running the benchmark locally or in CI.

### Benchmarks Fail on CPU / GitHub Actions

- **Symptom**: Out-of-memory, CUDA errors, or timeouts.
- **Fixes**:
  - Force CPU: `export CUDA_VISIBLE_DEVICES=""` before running.
  - Use small models (≤ 3B parameters; preferably ≤ 1B for CI).
  - Use `--limit` and small `--batch-size` for smoke tests:
    ```bash
    python benchmarks/evaluation/run_benchmark.py \
      --submission-file models/submissions/tiny_test.yaml \
      --output-dir results/raw/ \
      --limit 5 \
      --batch-size 1
    ```

### Model Cannot Be Loaded

- **Symptom**: Errors from `transformers` / `HFLM` when loading the model.
- **Checklist**:
  - The `hf_repo` in your YAML is correct and public.
  - You can load the model with:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained("your/repo")
    AutoModelForCausalLM.from_pretrained("your/repo")
    ```
  - Remove or adjust exotic `trust_remote_code` models when running in locked‑down CI.

### Leaderboard Does Not Update After a Run

- Run ordering:
  1. Run the benchmark.
  2. Aggregate results (optional but recommended).
  3. Update the registry.
  4. Sync the website data.

- Example:
  ```bash
  python benchmarks/evaluation/run_benchmark.py \
    --submission-file models/submissions/tiny_test.yaml \
    --output-dir results/raw/ \
    --limit 5

  python scripts/generate_report.py \
    --results-dir results/raw/TinyStories-1M-Verify \
    --output results/processed/tinystories_report.json

  python scripts/update_registry.py \
    --submission models/submissions/tiny_test.yaml \
    --results results/processed/tinystories_report.json \
    --registry models/registry.json

  python scripts/update_website.py \
    --registry models/registry.json \
    --output website/assets/data/leaderboard.json
  ```

### Scores Look Suspicious

- Confirm the **scoring formula** in `README.md` matches what your code computes.
- Remember: hardware‑dependent metrics (latency, throughput, energy, CO₂) do **not** affect aggregate ranking.
- If a metric seems off:
  - Inspect the raw JSON under `results/raw/...`.
  - File an issue with a sample of the raw JSON and your environment details.






