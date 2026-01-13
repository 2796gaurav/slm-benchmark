
# Contributing to SLM Benchmark

Thank you for your interest in contributing! This guide will help you add your model to the benchmark.

## How to Submit a Model

### 1. Create a Model Configuration

Create a YAML file in `models/` directory:

```yaml
# models/your-model.yaml
name: "YourModel-1B"
hf_model_id: "your-org/your-model"
parameters: 1000000000  # Must be ≤3B
architecture: "transformer"
context_length: 2048
languages: ["en"]
quantization:
  - type: "none"  # FP16 baseline
tags: ["instruction-tuned"]
license: "Apache-2.0"
```

### 2. Submit a Pull Request

1. Fork the repository
2. Add your model config: `models/your-model.yaml`
3. Create a PR with title: "Add [YourModel] to benchmark"
4. Our validation bot will check your config
5. Once approved and merged, benchmarks run automatically

### 3. Wait for Results

- Benchmarks typically complete within 24-48 hours
- Results are auto-committed to `results/your-model/`
- Leaderboard updates automatically

## Requirements

- ✅ Model must be ≤3B parameters
- ✅ Model must be on HuggingFace Hub
- ✅ License must be open-source
- ✅ Model config must pass validation

## Need Help?

- 📖 See [FAQ](FAQ.md)
- 💬 Open an issue on GitHub
- 📧 Email: benchmark@slm.dev

## Code of Conduct

Be respectful and constructive. We're building this for the community!