
# 🚀 SLM Benchmark

**Unbiased, comprehensive benchmarking for Small Language Models (1M-3B parameters)**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Models](https://img.shields.io/badge/models-24-green.svg)](https://2796gaurav.github.io/slm-benchmark)
[![Contributors](https://img.shields.io/github/contributors/2796gaurav/slm-benchmark.svg)](https://github.com/2796gaurav/slm-benchmark/graphs/contributors)

[🌐 View Leaderboard](https://2796gaurav.github.io/slm-benchmark) | [📖 Docs](docs/) | [🤝 Contribute](docs/CONTRIBUTING.md)

## Features

- ✅ **Unbiased**: Automated testing, no manual intervention
- 📊 **Comprehensive**: 23+ benchmark tasks covering accuracy, speed, and safety
- 🔓 **Open**: Apache 2.0 licensed, full reproducibility
- 🚀 **Fast**: Results within 24-48 hours of submission
- 💰 **Free**: Zero infrastructure cost (GitHub + Colab free tiers)

## Quick Start

### Run Benchmarks Locally

```bash
# Install dependencies
pip install -r benchmark/requirements.txt

# Run benchmark
python benchmark/run_eval.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --tasks leaderboard \
  --device cuda \
  --output results/qwen2-0.5b
```

### Submit Your Model

1. Create `models/your-model.yaml`
2. Submit a PR
3. Wait for automated benchmarks
4. See results on [leaderboard](https://2796gaurav.github.io/slm-benchmark)

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

## Benchmark Tasks

| Category | Tasks |
|----------|-------|
| Language Understanding | MMLU-Pro, ARC-Challenge, HellaSwag |
| Reasoning | BBH, GPQA Diamond, GSM8K |
| Coding | HumanEval+, MBPP |
| Safety | TruthfulQA, ToxiGen |

## Citation

```bibtex
@misc{slmbenchmark2025,
  title={SLM Benchmark: Comprehensive Evaluation of Small Language Models},
  author={Gaurav and Contributors},
  year={2025},
  url={https://github.com/2796gaurav/slm-benchmark}
}
```

## License

Apache 2.0 (code) | CC BY 4.0 (data)