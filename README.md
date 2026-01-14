# SLM Benchmark

**A focused, transparent benchmark for Small Language Models (1M–3B parameters).**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Website](https://img.shields.io/badge/leaderboard-live-green.svg)](https://2796gaurav.github.io/
[![Star on GitHub](https://img.shields.io/github/stars/2796gaurav/slm-benchmark?style=social)](https://github.com/2796gaurav/slm-benchmark)slm-benchmark)

**View the Live Leaderboard:**  
https://2796gaurav.github.io/slm-benchmark

---

## Overview

This project provides a standardized evaluation framework specifically for Small Language Models (SLMs) up to ~3B parameters. Unlike general benchmarks, SLM Benchmark focuses on:

- **Hardware Neutrality:** Rankings are based purely on accuracy-style scores. Latency, energy, and CO₂ are reported for information but do not affect the ranking.
- **Transparency:** Every evaluation run produces inspectable JSON artifacts.
- **Safety First:** Safety and fairness are treated as a core scoring pillar, not a footnote.

## Methodology

### Evaluation Pillars

The benchmark evaluates models across five weighted categories using the `lm-evaluation-harness` backend:

1. **Reasoning (35%)**: MMLU, ARC-Challenge, HellaSwag, TruthfulQA  
2. **Coding (20%)**: HumanEval, MBPP (with safe execution)  
3. **Math (15%)**: GSM8K, Math QA  
4. **Language (20%)**: BoolQ, PIQA, WinoGrande  
5. **Safety (20%)**: Toxicity, bias, truthfulness, and fairness probes  

> Note: Long-context tasks and edge metrics (latency/memory) are recorded but excluded from the aggregate score.

### Scoring Formula

Each pillar is normalized to **[0, 100]**. The final rank is determined by:

```
Score_final = 0.35R + 0.20C + 0.15M + 0.20L + 0.20S
```

## How to Submit a Model

We welcome community submissions. A model must be:

- ≤ **3B parameters**
- **Public on Hugging Face**
- Released under a **permissive license**

### Option 1: GitHub Issue (Recommended)

1. Copy the YAML schema below  
2. Open a new issue using the **Model Submission** template  
   https://github.com/2796gaurav/slm-benchmark/issues  
3. Paste your YAML in the issue description  

#### Submission YAML Schema

```yaml
model:
  name: "SmolLM2-1.7B"
  family: "SmolLM"
  hf_repo: "HuggingFaceTB/SmolLM2-1.7B"
  parameters: "1.7B"
  architecture: "llama"
  context_length: 8192
  license: "Apache-2.0"

  categories:
    - reasoning
    - coding
    - math
    - language
    - safety

  submitted_by: "github_username"
```

### Option 2: Run Locally

Advanced users can run the benchmark locally and submit the raw results.

```bash
# Setup
git clone https://github.com/2796gaurav/slm-benchmark.git
cd slm-benchmark
pip install -r requirements.txt

# Execution
python benchmarks/evaluation/run_benchmark.py \
  --submission-file models/submissions/your_model.yaml \
  --output-dir results/raw/
```

## Documentation & Support

- **Contributing:** See `CONTRIBUTING.md`
- **Developer Guide:** See `DEVELOPER_GUIDE.md`
- **Support:** Open a GitHub Issue or Discussion

## License

Distributed under the **Apache 2.0 License**.

Powered by **EleutherAI (lm-evaluation-harness)** and **Hugging Face**.