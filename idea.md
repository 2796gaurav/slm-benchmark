# SLM Benchmark - Complete System Architecture & Implementation Plan

## Executive Summary

**SLM Benchmark** is an open-source, unbiased benchmarking platform for Small Language Models (1M-3B parameters) optimized for edge devices, quantized models, and resource-constrained environments. Built on proven frameworks (EleutherAI's lm-evaluation-harness) with automated testing, GitHub Pages deployment, and zero-cost operation.

---

## 1. System Architecture

### 1.1 Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Repository                     │
│         github.com/2796gaurav/SLMbenchmark              │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼───────┐  ┌───────▼────────┐
│   Benchmark    │  │   Results    │  │    Website     │
│    Runner      │  │   Storage    │  │  (GitHub Pages)│
│ (GitHub Actions│  │  (JSON/CSV)  │  │  Static HTML   │
│  + Colab)      │  │              │  │                │
└────────────────┘  └──────────────┘  └────────────────┘
```

### 1.2 Technology Stack

- **Benchmark Engine**: EleutherAI lm-evaluation-harness (industry standard)
- **Testing Environment**: GitHub Actions + Google Colab T4 (free tier)
- **Model Support**: HuggingFace Transformers, vLLM, GGUF (via llama.cpp)
- **Quantization**: GPTQ, AWQ, GGUF (4-bit/8-bit)
- **Frontend**: Static HTML/CSS/JS (React components in artifacts)
- **Data Storage**: Git-based (JSON files committed to repo)
- **Deployment**: GitHub Pages (https://2796gaurav.github.io/slm-benchmark)

---

## 2. Benchmark Categories (360° Evaluation)

### 2.1 Task Categories

| Category | Benchmarks | Purpose |
|----------|-----------|---------|
| **Language Understanding** | MMLU-Pro, ARC-Challenge, HellaSwag | Core comprehension |
| **Reasoning** | BBH, GPQA Diamond, GSM8K | Logic & math |
| **Coding** | HumanEval+, MBPP, CodeXGLUE | Programming ability |
| **Multilingual** | FLORES-200, XNLI, Belebele | Cross-language support |
| **Instruction Following** | IFEval, MT-Bench | User intent alignment |
| **Truthfulness** | TruthfulQA | Factual accuracy |
| **Safety** | ToxiGen, BOLD | Bias & toxicity |

### 2.2 Performance Metrics

| Metric | Description | Hardware |
|--------|-------------|----------|
| **Accuracy** | Task-specific scores | All |
| **Throughput** | Tokens/second | GPU, CPU, Edge |
| **Latency** | Time to first token (TTFT), Inter-token latency | All |
| **Memory** | VRAM/RAM usage (GB) | All |
| **Energy** | Power consumption (Watts), mWh/token | Edge devices |
| **Model Size** | Disk space (GB), quantized size | All |

### 2.3 Hardware Profiles

1. **Cloud GPU**: NVIDIA T4 (16GB VRAM) - Free Colab
2. **Edge Device**: Simulated via CPU benchmarks
3. **Quantized**: 4-bit GPTQ, 4-bit AWQ, Q4_K_M GGUF

---

## 3. Unbiased Design Principles

### 3.1 Transparency

- All code open-source (Apache 2.0 license)
- Automated testing via GitHub Actions (no manual intervention)
- Reproducible: Exact commit SHA, model version, hardware specs logged
- Public results: All JSON files in `/results` directory

### 3.2 Fairness

- Same hardware for all models (GitHub Actions runners + Colab T4)
- Same prompts, same few-shot examples (lm-eval-harness defaults)
- Same evaluation order (alphabetical by model name to avoid warmup bias)
- No cherry-picking: All runs saved, failures logged

### 3.3 Community Governance

- **PR-based submissions**: Anyone can submit a model for benchmarking
- **Approval workflow**: 
  1. User submits PR with model config (`models/<model_name>.yaml`)
  2. Automated validation checks (model exists, size <3B, passes smoke test)
  3. Maintainer reviews, approves
  4. GitHub Actions runs full benchmark
  5. Results auto-committed to `results/<model_name>/`
  6. Website auto-updates
- **Dispute resolution**: Issues on GitHub, public discussion

---

## 4. Implementation Details

### 4.1 Directory Structure

```
slm-benchmark/
├── .github/
│   └── workflows/
│       ├── benchmark.yml        # Main benchmark runner
│       ├── validate_pr.yml      # PR validation
│       └── deploy_website.yml   # Deploy to GitHub Pages
├── benchmark/
│   ├── run_eval.py             # Wrapper for lm-eval-harness
│   ├── config.yaml             # Task groups, hardware profiles
│   ├── requirements.txt
│   └── utils/
│       ├── quantize.py         # GPTQ/AWQ/GGUF conversion
│       ├── metrics.py          # Custom metric computation
│       └── report.py           # JSON → Markdown/CSV
├── models/
│   ├── qwen2-0.5b.yaml        # Model configs
│   ├── phi-3-mini.yaml
│   └── tinyllama-1.1b.yaml
├── results/
│   ├── qwen2-0.5b/
│   │   ├── metadata.json       # Model info, hardware, timestamp
│   │   ├── accuracy.json       # Per-task scores
│   │   ├── performance.json    # Latency, throughput
│   │   └── report.md
│   └── ...
├── website/
│   ├── index.html              # Main leaderboard
│   ├── model.html              # Per-model detail page
│   ├── css/
│   ├── js/
│   │   ├── leaderboard.js      # Load JSON, render tables
│   │   ├── charts.js           # Plotly.js visualizations
│   │   └── filters.js          # Filter/sort/search
│   └── data/
│       └── models.json         # Aggregated results
├── docs/
│   ├── CONTRIBUTING.md
│   ├── ADDING_MODELS.md
│   └── FAQ.md
├── scripts/
│   ├── aggregate_results.py    # Combine all results
│   └── validate_model.py       # Check model size, format
├── README.md
└── LICENSE
```

### 4.2 Model Configuration (YAML)

```yaml
# models/qwen2-0.5b.yaml
name: "Qwen2-0.5B"
hf_model_id: "Qwen/Qwen2-0.5B-Instruct"
parameters: 494000000  # Must be ≤3B
architecture: "transformer"
context_length: 32768
languages: ["en", "zh", "es", "fr", "de", "ja", "ko", "ar"]
quantization:
  - type: "none"       # FP16 baseline
  - type: "gptq"
    bits: 4
  - type: "awq"
    bits: 4
  - type: "gguf"
    quant: "Q4_K_M"
tags: ["multilingual", "edge-friendly", "quantized"]
license: "Apache-2.0"
```

### 4.3 GitHub Actions Workflow

```yaml
# .github/workflows/benchmark.yml
name: Benchmark Model

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model to benchmark'
        required: true
        type: string
  pull_request:
    paths:
      - 'models/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate model config
        run: python scripts/validate_model.py models/${{ inputs.model_name }}.yaml

  benchmark-t4:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Trigger Colab via Paperspace Gradient (workaround)
        env:
          GRADIENT_API_KEY: ${{ secrets.GRADIENT_API_KEY }}
        run: |
          # Alternative: Use GitHub-hosted runner with CPU
          # or integrate with Paperspace/RunPod API
          python benchmark/run_eval.py \
            --model ${{ inputs.model_name }} \
            --tasks leaderboard \
            --device cpu \
            --output results/${{ inputs.model_name }}/
      
      - name: Commit results
        run: |
          git config user.name "SLM Benchmark Bot"
          git config user.email "bot@slmbenchmark.dev"
          git add results/${{ inputs.model_name }}/
          git commit -m "Add benchmark results for ${{ inputs.model_name }}"
          git push

  update-website:
    needs: benchmark-t4
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Aggregate results
        run: python scripts/aggregate_results.py
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./website
```

---

## 5. Website Features

### 5.1 Main Leaderboard

- **Sortable table**: Click column headers to sort
- **Filters**: 
  - Model size (0-0.5B, 0.5B-1B, 1B-3B)
  - Quantization (FP16, 4-bit GPTQ, 4-bit AWQ, GGUF)
  - License (Apache, MIT, Llama, etc.)
  - Language support (English, Multilingual)
  - Tags (edge-friendly, coding-focused, etc.)
- **Search**: Fuzzy search by model name
- **Visualizations**:
  - Scatter: Accuracy vs. Throughput
  - Bar: Top 10 models per task
  - Radar: Multi-dimensional comparison

### 5.2 Model Detail Page

- Full task breakdown (all 23+ benchmark scores)
- Hardware performance (T4, CPU, Edge)
- Quantization impact analysis
- Download links (HuggingFace, GGUF files)
- Reproduction guide (exact command to run)

### 5.3 Interactive Features

- Compare up to 4 models side-by-side
- Export results as CSV
- Permalink to specific filter/sort state
- Dark mode toggle

---

## 6. Colab Integration (Free Tier Optimization)

### 6.1 Notebook Structure

```python
# slm_benchmark_colab.ipynb

# Cell 1: Setup
!pip install lm-eval[openai] torch transformers vllm

# Cell 2: Clone repo
!git clone https://github.com/EleutherAI/lm-evaluation-harness
%cd lm-evaluation-harness

# Cell 3: Run benchmark
!lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2-0.5B-Instruct,dtype=float16 \
    --tasks leaderboard \
    --device cuda \
    --batch_size 8 \
    --output_path /content/drive/MyDrive/slm-benchmark-results/ \
    --log_samples

# Cell 4: Upload to GitHub
# (Automated via GitHub API)
```

### 6.2 GPU Time Management

- **Free T4 limits**: ~15 hours/week
- **Strategy**:
  - Run 1-2 models per session (~2 hours each)
  - Queue system: PR submission adds to queue
  - Weekly batch: Process queue every Sunday

---

## 7. Cost Analysis (Zero Infrastructure Cost)

| Resource | Provider | Cost |
|----------|----------|------|
| GPU (T4) | Google Colab Free | $0 |
| CI/CD | GitHub Actions (2000 min/month) | $0 |
| Hosting | GitHub Pages | $0 |
| Storage | GitHub (1GB soft limit) | $0 |
| Domain | GitHub subdomain | $0 |

**Total monthly cost: $0** (within free tiers)

---

## 8. Initial Model Pool (Starter Set)

| Model | Parameters | License | Why Include |
|-------|-----------|---------|-------------|
| Qwen2-0.5B | 494M | Apache 2.0 | Smallest, multilingual |
| Qwen2.5-1.5B | 1.5B | Apache 2.0 | Best <2B model (Dec 2024) |
| Phi-3-mini | 3.8B | MIT | Microsoft, strong reasoning |
| TinyLlama-1.1B | 1.1B | Apache 2.0 | Popular baseline |
| Gemma-2B | 2B | Gemma License | Google |
| StableLM-3B | 3B | Apache 2.0 | Stability AI |
| MiniCPM-2B | 2.4B | Apache 2.0 | Chinese-English bilingual |
| Smollm-360M | 360M | Apache 2.0 | Ultra-lightweight |

---

## 9. Automation & Maintenance

### 9.1 Scheduled Jobs

- **Weekly benchmark**: Re-run all models to catch regressions
- **Monthly report**: Summary email to maintainers
- **Quarterly cleanup**: Archive old results (>6 months)

### 9.2 Monitoring

- GitHub Actions success rate dashboard
- Website uptime (via UptimeRobot free tier)
- Result integrity checks (schema validation)

### 9.3 Community Tools

- **CLI**: `slm-benchmark submit <model_yaml>`
- **VS Code extension**: Browse results in IDE
- **Discord bot**: Notify new benchmarks

---

## 10. Extensibility

### 10.1 Adding New Benchmarks

```python
# benchmark/tasks/custom_task.py
from lm_eval.api.task import Task

class MyCustomTask(Task):
    VERSION = 0
    DATASET_PATH = "custom/dataset"
    
    def has_training_docs(self):
        return False
    
    def has_validation_docs(self):
        return True
    
    def doc_to_text(self, doc):
        return doc["question"]
    
    def doc_to_target(self, doc):
        return doc["answer"]
```

### 10.2 Adding Hardware Profiles

```yaml
# benchmark/config.yaml
hardware:
  raspberry_pi_5:
    cpu: "ARM Cortex-A76 (2.4GHz)"
    ram: "8GB LPDDR4X"
    runner: "self-hosted"
    tags: ["edge", "arm"]
```

---

## 11. Next Steps (Post-Launch Roadmap)

**Phase 1 (Month 1-2)**: Core infrastructure
- Set up GitHub repo
- Implement benchmark runner
- Deploy basic website
- Benchmark initial 8 models

**Phase 2 (Month 3-4)**: Community features
- PR submission workflow
- Model comparison tool
- Mobile-responsive design
- Documentation site

**Phase 3 (Month 5-6)**: Advanced features
- Real edge device testing (Raspberry Pi)
- Energy consumption metrics
- Synthetic data generation for custom tasks
- API for programmatic access

**Phase 4 (Ongoing)**: Ecosystem growth
- Partner with model developers
- Research collaborations
- Conference presentations
- Academic paper (arxiv)

---

## 12. Success Metrics

- **Adoption**: 100+ models benchmarked in 6 months
- **Community**: 50+ contributors, 500+ GitHub stars
- **Citations**: 10+ research papers citing the benchmark
- **Traffic**: 5,000+ monthly unique visitors to website

---

## 13. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GitHub rate limits | Cache model downloads, use mirrors |
| Colab quota exhaustion | Multiple accounts, paid Colab Pro as backup |
| Storage limits | Compress results, archive to Zenodo |
| Malicious PRs | Sandboxed execution, manual approval |
| Bias accusations | Public methodology docs, external audits |

---

## Contact & Governance

- **Maintainer**: @2796gaurav
- **Contributors**: Open to all (see CONTRIBUTING.md)
- **License**: Apache 2.0 (code), CC BY 4.0 (data)
- **Code of Conduct**: Contributor Covenant v2.1

---

**This is a living document. Last updated: 2025-01-13**