
# Frequently Asked Questions

## General

**Q: Why only models ≤3B parameters?**  
A: We focus on small models that can run on edge devices, mobile phones, and low-cost hardware. Larger models have plenty of benchmarks already.

**Q: How are benchmarks run?**  
A: We use EleutherAI's lm-evaluation-harness on standardized hardware (GitHub Actions + Google Colab T4 free tier).

**Q: Can I run benchmarks locally?**  
A: Yes! Clone the repo and use `benchmark/run_eval.py`. See README for details.

**Q: How often are benchmarks updated?**  
A: Weekly automated re-runs + on-demand when new models are added.

## Submitting Models

**Q: My model is 3.2B parameters. Can I still submit?**  
A: Unfortunately no. The 3B limit is strict to maintain focus on truly small models.

**Q: Can I submit quantized models?**  
A: Yes! Add quantization configs in your YAML. We benchmark both FP16 and quantized versions.

**Q: My model is private/requires authentication. Can it be benchmarked?**  
A: No, only public models on HuggingFace are supported.

## Results

**Q: Why did my model score low?**  
A: SLM Benchmark uses challenging tasks. Even state-of-the-art models score <70% on some benchmarks.

**Q: Can I dispute results?**  
A: Yes! Open an issue with evidence. All benchmarks are reproducible.

**Q: How is the average score calculated?**  
A: Weighted average across key tasks (MMLU, ARC, HellaSwag, GSM8K, TruthfulQA).

## Technical

**Q: What hardware is used?**  
A: NVIDIA T4 (16GB VRAM) for GPU tests, standard GitHub Actions runners for CPU tests.

**Q: Can I add custom benchmarks?**  
A: Yes! Submit a PR with your task implementation. See developer docs.

**Q: Is the leaderboard biased?**  
A: We take unbiased evaluation seriously. All code is open-source, automated, and auditable.