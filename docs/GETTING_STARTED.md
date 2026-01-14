# Submit Your Model in 3 Minutes

## Step 1: Create YAML File (30 seconds)

Create `models/submissions/my-model.yaml`:

```yaml
model:
  hf_repo: "myusername/my-awesome-1b-model"
```

That's it! Everything else is auto-detected.

## Step 2: Test Locally (Optional - 2 minutes)

```bash
# Clone repo
git clone https://github.com/slm-benchmark/slm-benchmark
cd slm-benchmark

# Quick validation
python benchmarks/validation/auto_detector.py \
  --hf-repo myusername/my-awesome-1b-model

# Output:
# ✓ Detected: My-Awesome-1B-Model
# ✓ Parameters: 1.2B
# ✓ Architecture: LlamaForCausalLM
# ✓ Context: 4096 tokens
# ✓ Formats: FP16, INT8, GGUF
```

## Step 3: Submit PR (2 minutes)

1. Fork the repository
2. Add your YAML file to `models/submissions/`
3. Create a Pull Request

## What Happens Next?

1. **Auto-Validation** (instant): Bot posts detected model properties to your PR.
2. **Quick Test** (5 min): Maintainers run `/quick-test` to verify the model runs.
3. **Full Benchmark** (2 hours): If passed, full evaluation is triggered.
4. **Leaderboard**: Your model appears on [slm-benchmark.dev](https://slm-benchmark.dev) automatically.

## Understanding Scores

| Category | Weight | Description |
|----------|--------|-------------|
| **Reasoning** | 30% | Logic, world knowledge (MMLU, ARC) |
| **Coding** | 20% | Code generation (HumanEval, MBPP) |
| **Math** | 15% | Problem solving (GSM8K) |
| **Language** | 12% | Comprehension (BoolQ, PIQA) |
| **Edge** | 10% | Latency, throughput, memory |
| **Safety** | 5% | Bias, toxicity |
| **Tool Use** | 8% | Function calling capabilities |

### Efficiency & Environmental

We now track:
- **Carbon Footprint**: CO2 emissions per 1000 queries
- **Energy Efficiency**: Accuracy per kWh

Higher efficiency scores mean you get more intelligence for less power!
