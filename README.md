# ⚡ SLM Benchmark

**Comprehensive, Unbiased Benchmarking Platform for Small Language Models (1M-3B parameters)**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Pages](https://img.shields.io/badge/website-live-green.svg)](https://2796gaurav.github.io/slm-benchmark)

## 🎯 Mission

Provide the most comprehensive, transparent, and unbiased evaluation platform for Small Language Models, enabling researchers and developers to make informed decisions about model selection for edge devices and resource-constrained environments.

## ✨ Features

### 🔬 360° Evaluation
- **Reasoning**: MMLU, ARC-Challenge, HellaSwag, TruthfulQA
- **Coding**: HumanEval, MBPP
- **Math**: GSM8K, MATH
- **Language Understanding**: BoolQ, PIQA, WinoGrande
- **Edge Performance**: Latency, throughput, memory, energy efficiency
- **Quantization Quality**: Accuracy retention across Q4_K_M, Q8_0, FP16
- **Safety**: Toxicity, bias, truthfulness metrics
- **Long Context**: Context handling up to model limits

### 🤖 Fully Automated
- Zero manual intervention in testing
- Deterministic, reproducible results
- GitHub Actions CI/CD pipeline
- Automated PR validation and testing
- Real-time leaderboard updates

### 🔒 Unbiased & Transparent
- All evaluation code open source
- Fixed random seeds for reproducibility
- Identical testing environment for all models
- Complete testing history in Git
- No cherry-picking or selective reporting

### 🌐 Beautiful Web Interface
- Interactive leaderboard with filters
- Detailed model comparison
- Performance visualizations
- Responsive design for all devices

## 🚀 Quick Start

### For Users: View Results

Visit **[https://2796gaurav.github.io/slm-benchmark](https://2796gaurav.github.io/slm-benchmark)** to browse the leaderboard.

### For Contributors: Submit a Model

1. **Fork the repository**
   ```bash
   git clone https://github.com/2796gaurav/slm-benchmark.git
   cd slm-benchmark
   ```

2. **Create your model submission file**
   ```bash
   cp models/submissions/template.yaml models/submissions/your-model.yaml
   ```

3. **Fill in model details** (see [Submission Format](#submission-format) below)

4. **Create a Pull Request**
   - The validation bot will automatically check your submission
   - If valid, maintainer will trigger `/test-benchmark`
   - Results will be published to the leaderboard

## 📝 Submission Format

Create a YAML file in `models/submissions/` with this structure:

```yaml
model:
  name: "SmolLM2-1.7B"
  family: "SmolLM"
  version: "2.0"
  
  # Hugging Face details
  hf_repo: "HuggingFaceTB/SmolLM2-1.7B"
  hf_revision: "main"  # or specific commit hash
  
  # Model specifications
  parameters: "1.7B"  # Must be ≤3B
  architecture: "llama"
  context_length: 8192
  license: "Apache-2.0"
  
  # Quantizations to test
  quantizations:
    - name: "FP16"
      format: "safetensors"
    - name: "Q4_K_M"
      format: "gguf"
      source: "bartowski/SmolLM2-1.7B-GGUF"
    - name: "Q8_0"
      format: "gguf"
      source: "bartowski/SmolLM2-1.7B-GGUF"
  
  # Categories (select all that apply)
  categories:
    - "reasoning"
    - "coding"
    - "edge-optimized"
  
  # Submission metadata
  submitted_by: "github_username"
  submitted_date: "2026-01-13"
  contact: "email@example.com"
  
  # Testing preferences (optional)
  testing:
    priority: "standard"  # standard | fast | comprehensive
    skip_safety: false
    edge_devices: ["cpu", "gpu"]
```

### Submission Requirements

✅ **Required:**
- Model must have ≤3B parameters
- Must be publicly accessible on Hugging Face
- Must have open license allowing benchmarking
- All required fields must be filled

❌ **Not Allowed:**
- Models >3B parameters
- Private or gated models
- Models with restrictive licenses
- Duplicate submissions

## 🔄 Automated Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Create PR with model YAML                          │
│  2. Validation Bot checks submission ✓                 │
│  3. Maintainer reviews & comments "/test-benchmark"    │
│  4. Testing Bot runs comprehensive benchmarks          │
│  5. Results posted as PR comment                       │
│  6. Maintainer reviews & comments "/push-results"      │
│  7. Publishing Bot updates leaderboard & merges PR     │
│  8. Website automatically rebuilds and deploys         │
└─────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture

### Repository Structure

```
slm-benchmark/
├── .github/
│   ├── workflows/           # GitHub Actions CI/CD
│   └── scripts/             # Automation scripts
├── benchmarks/
│   ├── evaluation/          # Benchmark runners
│   └── configs/             # Task configurations
├── models/
│   ├── registry.json        # All approved models
│   └── submissions/         # Pending submissions
├── results/
│   ├── raw/                 # Raw benchmark outputs
│   ├── processed/           # Aggregated results
│   └── archives/            # Historical snapshots
├── website/                 # GitHub Pages site
│   ├── index.html
│   └── assets/
└── scripts/                 # Utility scripts
```

### Tech Stack

- **Benchmarking**: EleutherAI's lm-evaluation-harness
- **Models**: Hugging Face Transformers, llama.cpp
- **CI/CD**: GitHub Actions
- **Website**: Vanilla HTML/CSS/JS (no build step needed)
- **Hosting**: GitHub Pages
- **Data**: JSON (no database needed)

## 📊 Benchmark Methodology

### Deterministic Testing
- Fixed random seed: `42`
- Controlled environment: Docker containers
- Identical hardware: Self-hosted runners with NVIDIA GPUs
- Reproducible results: All parameters logged

### Scoring System

**Aggregate Score** = Weighted average:
- Reasoning: 30%
- Coding: 20%
- Math: 15%
- Language: 15%
- Edge Performance: 10%
- Safety: 10%

Each category score is normalized to 0-100 scale.

### Hardware Specifications

All models tested on:
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: AMD Ryzen 9 5950X
- **RAM**: 64GB DDR4
- **Storage**: NVMe SSD

## 🔒 Security & Rate Limiting

To prevent abuse of GitHub Actions and GPU resources:

### Rate Limits
- **Per User**: 3 submissions per day
- **Per Model**: 1 benchmark run per day
- **Total**: 10 benchmarks per day across all users

### Access Control
- Only repository owner can trigger benchmarks
- PRs cannot modify workflow files or evaluation code
- All changes audited and logged

### Resource Protection
- Automatic cleanup after each run
- Cost estimation before running
- GPU time tracking and limits

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Submit Models
Follow the [Quick Start](#quick-start) guide above.

### Improve Benchmarks
- Propose new evaluation tasks
- Suggest improvements to existing tests
- Help validate results

### Enhance Website
- Improve UI/UX
- Add new visualizations
- Fix bugs

### Documentation
- Improve guides and tutorials
- Translate documentation
- Create video tutorials

**Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.**

## 📖 Documentation

- **[Submission Guide](docs/SUBMISSION_GUIDE.md)** - How to submit models
- **[Methodology](docs/BENCHMARK_METHODOLOGY.md)** - Detailed testing methodology
- **[Reproducibility](docs/REPRODUCIBILITY.md)** - How to reproduce results
- **[FAQ](docs/FAQ.md)** - Frequently asked questions
- **[API](docs/API.md)** - Programmatic access to results

## 🎯 Roadmap

### Phase 1: Launch (January 2026) ✅
- [x] Core benchmarking pipeline
- [x] Automated CI/CD workflows
- [x] Website with leaderboard
- [x] Initial model submissions

### Phase 2: Expansion (Q1 2026)
- [ ] Add 20+ models to leaderboard
- [ ] Multilingual evaluation suite
- [ ] Mobile/ARM device benchmarks
- [ ] Community voting system

### Phase 3: Advanced Features (Q2 2026)
- [ ] Real-world task benchmarks
- [ ] Custom evaluation workflows
- [ ] API for programmatic access
- [ ] Historical trend analysis

### Phase 4: Ecosystem (Q3-Q4 2026)
- [ ] Integration with model hubs
- [ ] Automated model discovery
- [ ] Benchmark-as-a-service
- [ ] Research partnerships

## 🌟 Starter Models

We've pre-benchmarked these models to get you started:

### HuggingFace
- **SmolLM2-1.7B** - Latest SLM from HuggingFace
- **SmolLM2-360M** - Efficient 360M parameter model
- **SmolLM2-135M** - Tiny but capable

### Qwen
- **Qwen2.5-1.5B** - Strong reasoning abilities
- **Qwen2.5-0.5B** - Ultra-efficient edge model

### Microsoft
- **Phi-2** - 2.7B parameter model with strong performance

### Alibaba
- **MiniCPM-2B** - Competitive Chinese-English model

[View all benchmarked models →](https://2796gaurav.github.io/slm-benchmark)

## 📊 Sample Results

| Rank | Model | Params | Aggregate | Reasoning | Coding | Math |
|------|-------|--------|-----------|-----------|--------|------|
| 🥇 | Phi-2 | 2.7B | 74.2 | 68.5 | 52.3 | 48.7 |
| 🥈 | SmolLM2-1.7B | 1.7B | 71.8 | 65.2 | 48.9 | 45.3 |
| 🥉 | Qwen2.5-1.5B | 1.5B | 70.5 | 64.8 | 47.1 | 44.6 |

*Scores are normalized 0-100. Updated: January 2026*

## 🔗 Links

- **Website**: [https://2796gaurav.github.io/slm-benchmark](https://2796gaurav.github.io/slm-benchmark)
- **Repository**: [https://github.com/2796gaurav/slm-benchmark](https://github.com/2796gaurav/slm-benchmark)
- **Issues**: [Report bugs or request features](https://github.com/2796gaurav/slm-benchmark/issues)
- **Discussions**: [Join the conversation](https://github.com/2796gaurav/slm-benchmark/discussions)

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EleutherAI** - For lm-evaluation-harness
- **Hugging Face** - For Transformers and model hosting
- **ggerganov** - For llama.cpp
- **Community Contributors** - For model submissions and improvements

## 📧 Contact

- **Maintainer**: @2796gaurav
- **Email**: [Create an issue](https://github.com/2796gaurav/slm-benchmark/issues)
- **Twitter**: Coming soon

---

<div align="center">

**⚡ Built with ❤️ for the SLM community**

[![Star on GitHub](https://img.shields.io/github/stars/2796gaurav/slm-benchmark?style=social)](https://github.com/2796gaurav/slm-benchmark)

</div>