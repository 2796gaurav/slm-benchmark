# How to Submit Models to SLM Marketplace

## Quick Start

### 1. Create Submission File

```bash
cp models/submissions/template.yml models/submissions/my-model.yml
```

Edit `my-model.yml`:

```yaml
model:
  name: "MyModel-1.5B"
  family: "MyFamily"
  hf_repo: "username/MyModel-1.5B"
  parameters: "1.5B"  # Must be 1M-5B
  architecture: "llama"
  context_length: 8192
  license: "Apache-2.0"
  
  quantizations:
    - name: "FP16"
      format: "safetensors"
  
  categories:
    - "reasoning"
    - "coding"
  
  submitted_by: "your-github-username"
  submitted_date: "2026-01-15"
  contact: "your-email@example.com"
```

### 2. Validate Locally (Optional)

```bash
python .github/scripts/validate_submission.py \
  --submission-file models/submissions/my-model.yml
```

### 3. Create Pull Request

```bash
git checkout -b submit-my-model
git add models/submissions/my-model.yml
git commit -m "Submit MyModel-1.5B"
git push origin submit-my-model
```

Open PR on GitHub.

### 4. Wait for Auto-Validation

The workflow automatically:
- Detects model properties from HuggingFace
- Validates submission
- Posts results as PR comment

### 5. Maintainer Runs Tests

Maintainer comments on PR:
- `/quick-test` - Fast smoke test (~5 min)
- `/full-benchmark` - Complete evaluation (~2 hours)
- `/publish-results` - Add to marketplace

### 6. Model Published

Once published:
- Added to `models/registry.json`
- Appears on marketplace website
- PR merged automatically

---

## Submission Workflow Details

### Stage 1: Auto-Validation (FREE, Instant)

**Trigger:** PR opened with YAML file

**What happens:**
1. `auto_detector.py` downloads `config.json` from HuggingFace
2. Extracts: parameters, architecture, context_length, quantizations
3. Validates submission format
4. Posts auto-detected info as PR comment

**No benchmarking yet** - just validation.

### Stage 2: Quick Test (Maintainer Command)

**Command:** `/quick-test` (comment on PR)

**What happens:**
1. Runs `run_benchmark.py` with `--limit 100`
2. Tests core categories only (reasoning, coding, math, language)
3. Takes ~5 minutes
4. Posts results as PR comment

**Purpose:** Verify model is runnable before full benchmark.

### Stage 3: Full Benchmark (Maintainer Command)

**Command:** `/full-benchmark` (comment on PR)

**What happens:**
1. Runs complete benchmark suite:
   - Reasoning (MMLU, ARC, HellaSwag, TruthfulQA)
   - Coding (HumanEval, MBPP)
   - Math (GSM8K, MathQA)
   - Language (BoolQ, PIQA, WinoGrande)
   - Safety (toxicity, bias, truthfulness)
   - Long context
   - Edge performance (CPU metrics)
2. Enables carbon tracking
3. Takes ~2 hours
4. Saves results to `results/raw/`
5. Generates processed report
6. Posts summary as PR comment

**Results saved:**
- `results/raw/{model_name}/{timestamp}/{model}_{quant}_{timestamp}.json`
- `results/processed/latest_benchmark.json`

### Stage 4: Publish Results (Maintainer Command)

**Command:** `/publish-results` (comment on PR)

**What happens:**
1. Downloads benchmark artifacts
2. Runs `update_registry.py`:
   - Loads submission YAML
   - Loads benchmark results JSON
   - Maps results to use_cases
   - Calculates aggregate score
   - Creates model entry
   - Updates `models/registry.json`
3. Runs `update_website.py`:
   - Copies registry to `website/assets/data/leaderboard.json`
4. Commits changes to main
5. Merges PR automatically

**Model now appears on marketplace!**

---

## Benchmark Categories

### Current Benchmarks

1. **Reasoning** (35% weight)
   - MMLU (Massive Multitask Language Understanding)
   - ARC-Challenge (AI2 Reasoning Challenge)
   - HellaSwag (Commonsense reasoning)
   - TruthfulQA (Truthfulness)

2. **Coding** (20% weight)
   - HumanEval (Python code generation)
   - MBPP (Mostly Basic Python Problems)

3. **Math** (15% weight)
   - GSM8K (Grade school math)
   - MathQA (Math word problems)

4. **Language** (20% weight)
   - BoolQ (Boolean questions)
   - PIQA (Physical interaction QA)
   - WinoGrande (Coreference resolution)

5. **Safety** (20% weight)
   - Toxicity detection
   - Bias measurement
   - Truthfulness

### Planned Benchmarks (Not Yet Implemented)

1. **RAG**
   - NIAH (Needle-in-Haystack)
   - RULER
   - RAGTruth
   - FRAMES

2. **Function Calling**
   - BFCL (Berkeley Function Calling Leaderboard)
   - NESTful
   - DispatchQA

3. **Domain-Specific**
   - MultiMedQA (healthcare)
   - FinBen (finance)
   - LegalBench (legal)

---

## Registry Schema

### Model Entry Structure

```json
{
  "id": "mymodel-1.5b",
  "name": "MyModel-1.5B",
  "family": "MyFamily",
  "hf_repo": "username/MyModel-1.5B",
  "parameters": "1.5B",
  "license": "Apache-2.0",
  "context_window": 8192,
  
  "use_cases": {
    "rag": {
      "score": 87.3,
      "benchmarks": {
        "niah": 87.3,
        "ruler": 84.1
      },
      "recommended": true
    },
    "function_calling": {
      "score": 72.1,
      "benchmarks": {
        "bfcl_v2": 72.1
      },
      "recommended": false
    }
  },
  
  "performance": {
    "hardware": "GitHub Actions (2-core CPU)",
    "quantizations": {
      "fp16": {
        "tps_output": 24.1,
        "ttft_ms": 187,
        "ram_gb": 6.2
      }
    }
  },
  
  "safety": {
    "guardrails_compatible": true,
    "hallucination_rate": 3.2,
    "bias_score": "B+",
    "toxicity_rate": 1.1
  },
  
  "aggregate_score": 74.3,
  "rank": 1,
  "date_added": "2026-01-15",
  "submitted_by": "github-username"
}
```

---

## Requirements

### Model Requirements

- **Parameters:** 1M-5B (1,000,000 to 5,000,000,000)
- **License:** Must allow benchmarking
- **HuggingFace:** Model must be publicly accessible
- **Format:** Must have `config.json` and model weights

### Submission Requirements

- **YAML Format:** Follow `template.yml`
- **Required Fields:**
  - `name`, `family`, `hf_repo`, `parameters`
  - `architecture`, `license`, `submitted_by`
- **Quantizations:** At least one (FP16 minimum)
- **Categories:** At least one

### Validation Checks

The validation script checks:
- ✅ Parameter count 1M-5B
- ✅ HuggingFace repo exists and is accessible
- ✅ Required fields present
- ✅ Quantization formats valid
- ✅ No duplicate models in registry

---

## Troubleshooting

### Common Issues

1. **"Model exceeds 5B limit"** or **"Model below 1M minimum"**
   - Check `parameters` field in YAML
   - Format: "1.5B", "2.7B", etc.

2. **"Cannot access HuggingFace repo"**
   - Verify repo is public
   - Check repo ID format: `username/model-name`
   - Ensure model has `config.json`

3. **"Validation failed"**
   - Check YAML syntax
   - Ensure all required fields are present
   - Verify field values are not empty

4. **"Benchmark failed"**
   - Model may not support required tasks
   - Check model compatibility with lm-eval
   - Verify model loads correctly

### Getting Help

- **GitHub Issues:** Open issue on repository
- **PR Comments:** Ask questions on your PR
- **Documentation:** See `CODEBASE_ANALYSIS.md` for technical details

---

## Code Locations

### Key Files

- **Template:** `models/submissions/template.yml`
- **Validation:** `.github/scripts/validate_submission.py`
- **Auto-detection:** `benchmarks/validation/auto_detector.py`
- **Workflow:** `.github/workflows/02-submission-workflow.yml`
- **Benchmark:** `benchmarks/evaluation/run_benchmark.py`
- **Registry Update:** `scripts/update_registry.py`
- **Registry:** `models/registry.json`

### Understanding the Code

See `CODEBASE_ANALYSIS.md` for:
- Complete architecture overview
- Detailed code explanations
- Benchmark pipeline logic
- Registry update process

See `IMPLEMENTATION_GUIDE.md` for:
- Missing feature implementations
- Code examples
- Integration guides

---

## Best Practices

1. **Test Locally First**
   - Validate YAML before PR
   - Test model loading if possible

2. **Provide Complete Information**
   - Fill all fields in template
   - Add relevant categories
   - Specify quantizations

3. **Be Patient**
   - Full benchmark takes ~2 hours
   - Maintainers review manually
   - Results are posted as PR comments

4. **Follow Up**
   - Check PR comments for results
   - Address any issues maintainers find
   - Ask questions if unclear

---

## Next Steps

After your model is published:

1. **Share Results**
   - Link to marketplace entry
   - Share on social media
   - Update model card on HuggingFace

2. **Monitor Performance**
   - Check ranking on marketplace
   - Compare with other models
   - Track improvements

3. **Contribute**
   - Submit improvements
   - Report bugs
   - Suggest new benchmarks

---

## Summary

**Submission is simple:**
1. Create YAML file
2. Open PR
3. Wait for auto-validation
4. Maintainer runs tests
5. Model published!

**Full process takes:**
- Auto-validation: ~2 minutes
- Quick test: ~5 minutes (if requested)
- Full benchmark: ~2 hours (if requested)
- Publishing: ~5 minutes

**Total time:** ~2-3 hours (mostly waiting for benchmarks)
