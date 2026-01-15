# SLM Marketplace 🏪

**The definitive platform for discovering and deploying Small Language Models for Edge AI.**

Find the perfect model for your use case—RAG, function calling, coding, or domain-specific tasks—all benchmarked on real CPU hardware.

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue)](https://2796gaurav.github.io/slm-benchmark/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Verified%20Models-5+-cyan)]()

## 🎯 What Makes Us Different

| Traditional Benchmarks | SLM Marketplace |
|----------------------|-----------------|
| Rank models 1-N | **Discover** by use case |
| Single aggregate score | **Multi-dimensional** profiling |
| GPU-focused | **CPU-first** benchmarking |
| "Best model" | "Best model **for YOUR needs**" |

## 🔍 Use-Case Evaluation

We evaluate models on what actually matters for edge deployment:

- **📚 RAG & Retrieval** - Needle-in-haystack, context utilization, faithfulness
- **⚡ Function Calling** - Single/multi/parallel calls, API accuracy
- **💻 Coding** - Code generation, completion, problem-solving
- **🧠 Reasoning** - General knowledge, logic, commonsense
- **🛡️ Safety & Guardrails** - Toxicity, bias, hallucination detection

## 📊 CPU Performance Metrics

All benchmarks run on GitHub Actions (2-core CPU), measuring:

- **TPS** - Tokens Per Second (output generation)
- **TTFT** - Time to First Token (latency)
- **RAM** - Peak memory usage
- **Size** - Model size by quantization (FP16, Q8, Q4)

> ⚠️ GPU inference may be 5-20x faster. CPU metrics are for relative comparison only.

## 🚀 Quick Start

### Browse Models
Visit [**SLM Marketplace**](https://2796gaurav.github.io/slm-benchmark/) to:
1. Select your use case (RAG, Function Calling, Coding, etc.)
2. Filter by RAM constraints and context length
3. Compare models side-by-side
4. View detailed model cards with performance data

### Submit a Model
1. Fork this repository
2. Create `models/submissions/your-model.yaml`:
   ```yaml
   model:
     hf_repo: "your-org/your-model"
   ```
3. Open a Pull Request
4. Maintainers will run benchmarks and publish results

## 📁 Project Structure

```
├── website/              # Static GitHub Pages site
│   ├── index.html        # Discovery homepage
│   ├── model.html        # Model detail page
│   ├── compare.html      # Side-by-side comparison
│   ├── methodology.html  # Benchmarking methodology
│   └── assets/
│       ├── data/         # leaderboard.json
│       ├── js/           # marketplace.js, main.js
│       └── css/          # style.css
├── models/
│   ├── registry.json     # Model registry (source of truth)
│   └── submissions/      # Model submission YAMLs
├── benchmarks/
│   └── evaluation/       # Benchmark runners
├── scripts/              # Processing scripts
└── .github/workflows/    # CI/CD pipelines
```

## 🔧 Local Development

```bash
# Clone the repo
git clone https://github.com/2796gaurav/slm-benchmark.git
cd slm-benchmark

# Serve website locally
cd website
python -m http.server 8080
# Open http://localhost:8080
```

## 📈 Supported Models (1M-5B)

| Tier | Example Models |
|------|---------------|
| **Ultra-Small** (< 100M) | Doge-20M, TinyStories-1M |
| **Tiny** (100M-500M) | SmolLM2-135M/360M, Qwen2.5-0.5B |
| **Small** (500M-1.5B) | SmolLM2-1.7B, Llama-3.2-1B, Qwen2.5-1.5B |
| **Medium** (1.5B-3B) | Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2.6B |
| **Large** (3B-5B) | Phi-3.5-Mini-3.8B, Gemma3-4B, Qwen2.5-4B |

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **Add a model**: Submit a PR with a YAML file
- **Improve benchmarks**: Enhance evaluation methodology
- **Fix bugs**: Check open issues

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the Edge AI community**
