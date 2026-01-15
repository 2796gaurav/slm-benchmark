# SLM Marketplace - Complete Codebase Analysis & Implementation Guide

## Executive Summary

This document provides a **deep technical analysis** of the SLM Marketplace codebase, explaining how everything works from actual code (not documentation), and what needs to be implemented to achieve the full marketplace vision.

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Model Submission Flow (From Code)](#model-submission-flow-from-code)
3. [Benchmarking Pipeline (Detailed Logic)](#benchmarking-pipeline-detailed-logic)
4. [Registry & Marketplace Schema](#registry--marketplace-schema)
5. [What's Implemented vs Missing](#whats-implemented-vs-missing)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Current Architecture Overview

### File Structure Analysis

```
SLMBenchmark/
├── models/
│   ├── registry.json          # Main marketplace registry (schema v2.0.0)
│   └── submissions/
│       └── template.yml       # Submission template
├── benchmarks/
│   ├── evaluation/
│   │   ├── run_benchmark.py   # Main benchmark orchestrator
│   │   ├── edge_benchmark.py  # CPU performance metrics
│   │   ├── safety_eval.py     # Safety & bias evaluation
│   │   ├── long_context_eval.py
│   │   └── ...
│   ├── validation/
│   │   └── auto_detector.py   # Auto-detect model properties
│   └── configs/
│       └── benchmark_tasks.yaml
├── scripts/
│   ├── update_registry.py      # Updates registry from results
│   ├── generate_report.py      # Aggregates benchmark results
│   └── update_website.py       # Syncs registry to website
├── website/
│   ├── index.html              # Marketplace homepage
│   ├── model.html              # Model detail page
│   └── assets/
│       ├── js/
│       │   ├── marketplace.js  # Marketplace filtering logic
│       │   └── leaderboard.js  # Legacy leaderboard
│       └── data/
│           └── leaderboard.json # Website data (copy of registry)
└── .github/
    ├── workflows/
    │   └── 02-submission-workflow.yml  # PR-based submission
    └── scripts/
        └── validate_submission.py      # Validates YAML submissions
```

---

## Model Submission Flow (From Code)

### Step-by-Step Process

#### **Step 1: Create Submission YAML**

**File:** `models/submissions/template.yml`

```yaml
model:
  name: "SmolLM2-1.7B"
  family: "SmolLM"
  hf_repo: "HuggingFaceTB/SmolLM2-1.7B"
  parameters: "1.7B"  # Must be ≤3B
  architecture: "llama"
  context_length: 8192
  license: "Apache-2.0"
  
  quantizations:
    - name: "FP16"
      format: "safetensors"
    - name: "Q8"
      format: "gguf"
  
  categories:
    - "reasoning"
    - "coding"
  
  submitted_by: "github-username"
  submitted_date: "2026-01-13"
  contact: "email@example.com"
```

**Validation Logic:** `.github/scripts/validate_submission.py`

```python
# Key validations:
- MAX_PARAMS = 3_000_000_000  # 3B limit
- Required fields: name, family, hf_repo, parameters, architecture, license
- Checks HuggingFace repo exists and is accessible
- Validates quantization formats
- Checks for duplicate models in registry
```

#### **Step 2: Open Pull Request**

**Workflow:** `.github/workflows/02-submission-workflow.yml`

**Stage 1: Auto-Validation (FREE, Instant)**
```yaml
# Triggers on PR open
auto-validate:
  - Runs auto_detector.py to detect model properties
  - Posts auto-detected info as PR comment
  - No benchmarking yet (just validation)
```

**Auto-Detection Logic:** `benchmarks/validation/auto_detector.py`

```python
def detect_all_properties(self, hf_repo: str):
    # Downloads config.json from HuggingFace
    # Extracts:
    #   - parameters (from config dimensions)
    #   - architecture
    #   - context_length (max_position_embeddings)
    #   - quantizations (scans repo files)
    #   - vocab_size, hidden_size, num_layers
```

#### **Step 3: Quick Test (Maintainer Command)**

**Command:** `/quick-test` (comment on PR)

**Logic:**
```yaml
quick-test:
  - Runs run_benchmark.py with --limit 100
  - Only tests core categories (reasoning, coding, math, language)
  - Takes ~5 minutes
  - Posts results as PR comment
```

**Implementation:** `benchmarks/evaluation/run_benchmark.py`

```python
# Quick test runs:
python benchmarks/evaluation/run_benchmark.py \
  --submission-file "$SUBMISSION_FILE" \
  --limit 100 \  # Only 100 samples per task
  --output-dir quick_results/
```

#### **Step 4: Full Benchmark (Maintainer Command)**

**Command:** `/full-benchmark` (comment on PR)

**Logic:**
```yaml
full-benchmark:
  - Runs complete benchmark suite
  - All categories, all samples
  - Enables carbon tracking
  - Takes ~2 hours
  - Saves results to results/raw/
  - Generates processed report
```

**Full Benchmark Execution:**

```python
# benchmarks/evaluation/run_benchmark.py

class SLMBenchmark:
    def run_full_benchmark(self):
        # 1. Load model
        model = self.load_model()  # Uses HFLM from lm_eval
        
        # 2. Run all benchmarks
        self.results['reasoning_scores'] = self.run_reasoning_benchmarks(model)
        self.results['coding_scores'] = self.run_coding_benchmarks(model)
        self.results['math_scores'] = self.run_math_benchmarks(model)
        self.results['language_scores'] = self.run_language_benchmarks(model)
        self.results['edge_metrics'] = self.run_edge_benchmarks(model)
        self.results['safety_scores'] = self.run_safety_benchmarks(model)
        self.results['long_context_scores'] = self.run_long_context_benchmarks(model)
        
        # 3. Calculate aggregate score
        aggregate_score = self.calculate_aggregate_score()
        
        # 4. Save results
        self.save_results(result, output_dir)
```

**Benchmark Categories:**

1. **Reasoning** (`run_reasoning_benchmarks`)
   - Tasks: `mmlu`, `arc_challenge`, `hellaswag`, `truthfulqa_mc2`
   - Uses: `lm_eval.evaluator.simple_evaluate()`
   - Weight: 35% in aggregate

2. **Coding** (`run_coding_benchmarks`)
   - Tasks: `humaneval`, `mbpp`
   - Batch size: 1 (required for code execution)
   - Weight: 20% in aggregate

3. **Math** (`run_math_benchmarks`)
   - Tasks: `gsm8k`, `math_qa`
   - Weight: 15% in aggregate

4. **Language** (`run_language_benchmarks`)
   - Tasks: `boolq`, `piqa`, `winogrande`
   - Weight: 20% in aggregate

5. **Safety** (`run_safety_benchmarks`)
   - Toxicity detection (Detoxify or keyword-based)
   - Bias measurement (simplified)
   - Weight: 20% in aggregate

6. **Edge Performance** (`run_edge_benchmarks`)
   - **NOT included in ranking** (hardware-dependent)
   - Measures: latency, throughput, memory, energy
   - Reported for transparency only

#### **Step 5: Publish Results (Maintainer Command)**

**Command:** `/publish-results` (comment on PR)

**Logic:**
```yaml
publish:
  - Downloads benchmark artifacts
  - Runs update_registry.py to add model to registry
  - Runs update_website.py to sync to website
  - Commits changes to main
  - Merges PR
```

**Registry Update:** `scripts/update_registry.py`

```python
def update_registry(submission_file, results_file, registry_file):
    # 1. Load submission YAML
    submission = yaml.safe_load(submission_file)
    
    # 2. Load benchmark results JSON
    results = json.load(results_file)
    
    # 3. Build use_cases from results
    use_cases = {}
    for uc in ['rag', 'function_calling', 'coding', 'reasoning', 'guardrails']:
        uc_results = results.get(f'{uc}_scores', {})
        if uc_results:
            score = _average_score(uc_results)  # Averages lm-eval metrics
            use_cases[uc] = {
                'score': score,
                'benchmarks': uc_results,
                'recommended': score >= 70.0
            }
    
    # 4. Build performance metrics
    performance = results.get('performance', {})
    
    # 5. Calculate aggregate score (weighted)
    weights = {
        'rag': 0.25,
        'function_calling': 0.20,
        'coding': 0.15,
        'reasoning': 0.25,
        'guardrails': 0.15
    }
    aggregate_score = sum(
        use_cases.get(uc, {}).get('score', 0) * weights.get(uc, 0)
        for uc in weights.keys()
    )
    
    # 6. Create model entry
    model_entry = {
        'id': model['name'].lower().replace(' ', '-'),
        'name': model['name'],
        'family': model['family'],
        'hf_repo': model['hf_repo'],
        'parameters': model['parameters'],
        'use_cases': use_cases,
        'performance': performance,
        'safety': safety,
        'aggregate_score': aggregate_score,
        ...
    }
    
    # 7. Add/update in registry
    registry['models'].append(model_entry)
    registry['models'].sort(key=lambda x: x['aggregate_score'], reverse=True)
    
    # 8. Save registry
    json.dump(registry, registry_file)
```

---

## Benchmarking Pipeline (Detailed Logic)

### Core Benchmark Runner

**File:** `benchmarks/evaluation/run_benchmark.py`

#### **1. Model Loading**

```python
def load_model(self):
    # Uses lm_eval's HFLM wrapper
    model = HFLM(
        pretrained=self.config.hf_repo,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        trust_remote_code=True,
        dtype='float16' if quantization == 'FP16' else None
    )
    return model
```

**Note:** Currently loads via HuggingFace transformers, not quantized formats. Quantization metadata is recorded but doesn't affect evaluation backend.

#### **2. Score Calculation**

```python
def _average_scores(self, score_dict: Dict) -> float:
    """Extracts accuracy from lm-eval results"""
    values = []
    for v in score_dict.values():
        if isinstance(v, dict):
            # lm-eval returns: {'acc_norm,none': 0.75, ...}
            acc_norm = v.get('acc_norm,none', v.get('acc_norm'))
            acc = v.get('acc,none', v.get('acc'))
            
            if acc_norm is not None:
                values.append(float(acc_norm) * 100)
            elif acc is not None:
                values.append(float(acc) * 100)
    
    return sum(values) / len(values) if values else 0.0
```

#### **3. Aggregate Score**

```python
def calculate_aggregate_score(self) -> float:
    weights = {
        'reasoning': 0.35,
        'coding': 0.20,
        'math': 0.15,
        'language': 0.20,
        'safety': 0.20,
    }
    
    scores = {
        'reasoning': self._average_scores(self.results.get('reasoning_scores', {})),
        'coding': self._average_scores(self.results.get('coding_scores', {})),
        'math': self._average_scores(self.results.get('math_scores', {})),
        'language': self._average_scores(self.results.get('language_scores', {})),
        'safety': self._average_scores(self.results.get('safety_scores', {})),
    }
    
    aggregate = sum(scores[k] * weights[k] for k in weights.keys())
    return aggregate
```

**Important:** Edge metrics, efficiency, and carbon are **NOT** included in aggregate (hardware-dependent).

### Edge Performance Benchmarking

**File:** `benchmarks/evaluation/edge_benchmark.py`

#### **Current Implementation:**

```python
class EdgeBenchmark:
    def measure_latency(self):
        # Measures inference latency (mean, p50, p95, p99)
        # Hardware-dependent, NOT used for ranking
        
    def measure_throughput(self):
        # Measures tokens per second
        # Hardware-dependent, NOT used for ranking
        
    def measure_memory(self):
        # Measures RAM/GPU memory usage
        # Hardware-dependent, NOT used for ranking
```

**Missing:** CPU-specific TPS/TTFT measurement as described in requirements.

---

## Registry & Marketplace Schema

### Current Registry Schema

**File:** `models/registry.json`

```json
{
  "models": [
    {
      "id": "qwen2.5-3b-instruct",
      "name": "Qwen2.5-3B-Instruct",
      "family": "Qwen",
      "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
      "parameters": "3B",
      
      "use_cases": {
        "rag": {
          "score": 87.3,
          "benchmarks": {...},
          "recommended": true
        },
        "function_calling": {
          "score": 72.1,
          "benchmarks": {...},
          "recommended": false
        }
      },
      
      "performance": {
        "hardware": "GitHub Actions (2-core CPU)",
        "quantizations": {
          "fp16": {
            "size_gb": 6.0,
            "ram_gb": 8.2,
            "tps_output": 12.1,
            "ttft_ms": 234
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
      "rank": 1
    }
  ],
  "metadata": {
    "last_updated": "2026-01-14T23:30:00Z",
    "total_models": 0,
    "schema_version": "2.0.0",
    "platform": "SLM Marketplace"
  }
}
```

### What's Implemented in Registry Update

**File:** `scripts/update_registry.py`

✅ **Implemented:**
- Basic use_cases structure (rag, function_calling, coding, reasoning, guardrails)
- Performance structure (placeholder)
- Safety structure (placeholder)
- Aggregate score calculation
- Ranking by aggregate score

❌ **Missing:**
- Domain-specific scores (finance, healthcare, legal)
- Detailed performance metrics (TPS, TTFT, memory per quantization)
- Deployment targets
- Compliance flags (HIPAA, GDPR, FINRA)
- Tags and best_for/not_for logic

---

## What's Implemented vs Missing

### ✅ Fully Implemented

1. **Submission Workflow**
   - YAML template
   - Validation script
   - Auto-detection
   - PR-based workflow
   - Maintainer commands (/quick-test, /full-benchmark, /publish-results)

2. **Core Benchmarking**
   - Reasoning (MMLU, ARC, HellaSwag, TruthfulQA)
   - Coding (HumanEval, MBPP)
   - Math (GSM8K, MathQA)
   - Language (BoolQ, PIQA, WinoGrande)
   - Safety (toxicity, bias, truthfulness)
   - Long context (placeholder)

3. **Registry System**
   - Basic marketplace schema
   - Use cases structure
   - Aggregate scoring
   - Ranking

4. **Website Frontend**
   - Marketplace homepage (index.html)
   - Model cards with filtering
   - Search functionality
   - Use case filters

### ⚠️ Partially Implemented

1. **Use Case Mapping**
   - **Current:** Maps benchmark categories to use cases in `update_registry.py`
   - **Missing:** 
     - RAG-specific benchmarks (NIAH, RULER, RAGTruth, FRAMES)
     - Function calling benchmarks (BFCL, NESTful, DispatchQA)
     - Guardrails benchmarks (jailbreak detection, prompt injection)

2. **Performance Metrics**
   - **Current:** Edge benchmark exists but measures generic latency/throughput
   - **Missing:**
     - CPU-specific TPS measurement (output generation)
     - TTFT measurement (time to first token)
     - Memory measurement per quantization
     - Context length testing (512, 2048, 8192)

3. **Registry Schema**
   - **Current:** Basic structure exists
   - **Missing:**
     - Domain scores (finance, healthcare, legal)
     - Detailed performance per quantization
     - Deployment targets
     - Compliance flags

### ❌ Not Implemented

1. **RAG Benchmarks**
   - NIAH (Needle-in-Haystack)
   - RULER
   - RAGTruth
   - FRAMES

2. **Function Calling Benchmarks**
   - Berkeley Function Calling Leaderboard (BFCL)
   - NESTful
   - DispatchQA

3. **Domain-Specific Benchmarks**
   - MultiMedQA (healthcare)
   - FinBen (finance)
   - LegalBench (legal)

4. **CPU Performance Benchmarking**
   - TPS measurement (tokens per second - output)
   - TTFT measurement (time to first token)
   - Memory measurement per quantization
   - Context length testing

5. **Model Detail Page**
   - Use case performance tabs
   - Performance charts (TPS, TTFT, Memory)
   - Deployment guide
   - Benchmark deep-dive

---

## Implementation Roadmap

### Phase 1: Complete Use Case Benchmarks

#### 1.1 RAG Benchmarks

**File to create:** `benchmarks/evaluation/rag_eval.py`

```python
class RAGEvaluator:
    def evaluate_all(self):
        results = {}
        
        # NIAH - Needle in Haystack
        results['niah'] = self._run_niah_test(
            context_lengths=[1024, 2048, 4096, 8192],
            document_depths=[0, 25, 50, 75, 100]
        )
        
        # RULER - Extended NIAH
        results['ruler'] = self._run_ruler_benchmark()
        
        # RAGTruth - Hallucination detection
        results['ragtruth'] = self._evaluate_hallucination()
        
        # FRAMES - Multi-hop reasoning
        results['frames'] = self._run_frames_benchmark()
        
        return results
```

**Integration:** Add to `run_benchmark.py`:

```python
def run_rag_benchmarks(self, model) -> Dict:
    from rag_eval import RAGEvaluator
    evaluator = RAGEvaluator(model, self.config)
    return evaluator.evaluate_all()
```

#### 1.2 Function Calling Benchmarks

**File to create:** `benchmarks/evaluation/function_calling_eval.py`

```python
class FunctionCallingEvaluator:
    def evaluate_all(self):
        results = {}
        
        # BFCL - Berkeley Function Calling Leaderboard
        results['bfcl_v2'] = self._run_bfcl_benchmark()
        
        # Test scenarios
        scenarios = [
            'single_function',
            'multiple_functions',
            'parallel_functions',
            'multi_turn'
        ]
        
        for scenario in scenarios:
            results[f'{scenario}_accuracy'] = self._test_function_scenario(scenario)
        
        return results
```

#### 1.3 Update Registry Mapping

**File:** `scripts/update_registry.py`

```python
# Update use_cases mapping
use_cases = {}
for uc in ['rag', 'function_calling', 'coding', 'reasoning', 'guardrails']:
    # Map benchmark results to use cases
    if uc == 'rag':
        uc_results = {
            'niah': results.get('rag_scores', {}).get('niah'),
            'ruler': results.get('rag_scores', {}).get('ruler'),
            'ragtruth': results.get('rag_scores', {}).get('ragtruth'),
            'frames': results.get('rag_scores', {}).get('frames')
        }
    elif uc == 'function_calling':
        uc_results = {
            'bfcl_v2': results.get('function_calling_scores', {}).get('bfcl_v2'),
            ...
        }
    # ... etc
```

### Phase 2: CPU Performance Benchmarking

#### 2.1 Create CPU Performance Module

**File to create:** `benchmarks/evaluation/cpu_performance.py`

```python
class CPUPerformanceBenchmark:
    """
    CPU-only performance benchmarking for GitHub Actions runners.
    
    Hardware Spec:
    - Platform: GitHub Actions Standard Runner
    - CPU: 2-core x86_64 @ 2.60GHz
    - RAM: 7 GB
    """
    
    HARDWARE_SPEC = {
        'platform': 'GitHub Actions Standard Runner',
        'cpu_cores': 2,
        'cpu_model': 'x86_64 @ 2.60GHz',
        'ram_gb': 7,
        'storage_gb': 14
    }
    
    def measure_tps(self, model, output_lengths=[100, 500, 1000], num_runs=5):
        """
        Measure Tokens Per Second (OUTPUT GENERATION ONLY)
        
        Args:
            model: Loaded model instance
            output_lengths: List of output token lengths to test
            num_runs: Number of runs per length
        
        Returns:
            Dict with TPS metrics per output length
        """
        results = {}
        
        for output_len in output_lengths:
            tps_values = []
            
            for run in range(num_runs):
                # Warm-up run
                if run == 0:
                    _ = model.generate("The meaning of life is", max_new_tokens=10)
                
                # Benchmark run
                prompt = "The meaning of life is"
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                start = time.perf_counter()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=output_len,
                    do_sample=False,  # Deterministic
                    pad_token_id=self.tokenizer.eos_token_id
                )
                end = time.perf_counter()
                
                # Calculate TPS
                elapsed = end - start
                tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
                tps = tokens_generated / elapsed
                
                tps_values.append(tps)
            
            results[f'{output_len}_tokens'] = {
                'mean_tps': np.mean(tps_values),
                'std_tps': np.std(tps_values),
                'median_tps': np.median(tps_values),
                'min_tps': np.min(tps_values),
                'max_tps': np.max(tps_values)
            }
        
        return results
    
    def measure_ttft(self, model, context_lengths=[512, 2048, 8192], num_runs=5):
        """
        Measure Time to First Token (Prefill Latency)
        
        Important for interactive applications.
        """
        results = {}
        
        for ctx_len in context_lengths:
            ttft_values = []
            
            for run in range(num_runs):
                # Generate context of specified length
                prompt = " ".join(["word"] * ctx_len)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len)
                
                # Measure time to first token
                start = time.perf_counter()
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,  # Just first token
                    do_sample=False
                )
                end = time.perf_counter()
                
                ttft_ms = (end - start) * 1000
                ttft_values.append(ttft_ms)
            
            results[f'{ctx_len}_context'] = {
                'mean_ttft_ms': np.mean(ttft_values),
                'std_ttft_ms': np.std(ttft_values),
                'median_ttft_ms': np.median(ttft_values)
            }
        
        return results
    
    def measure_memory(self, model):
        """
        Measure RAM usage during inference
        """
        import psutil
        process = psutil.Process()
        
        # Baseline memory
        baseline_mb = process.memory_info().rss / 1024 / 1024
        
        # Peak memory during generation
        prompt = "Tell me about AI"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        mem_before = process.memory_info().rss / 1024 / 1024
        _ = model.generate(**inputs, max_new_tokens=100)
        mem_after = process.memory_info().rss / 1024 / 1024
        
        return {
            'baseline_mb': baseline_mb,
            'model_loaded_mb': mem_before,
            'peak_generation_mb': mem_after,
            'overhead_mb': mem_after - baseline_mb,
            'ram_gb': mem_after / 1024  # Convert to GB
        }
    
    def run_full_benchmark(self, model, quantizations=['fp16', 'q8', 'q4']):
        """
        Run complete CPU performance benchmark for all quantizations
        """
        results = {}
        
        for quant in quantizations:
            logger.info(f"Testing quantization: {quant}")
            
            # Load quantized model (implementation depends on backend)
            quantized_model = self._load_quantized_model(model, quant)
            
            # Measure TPS
            tps_results = self.measure_tps(quantized_model)
            
            # Measure TTFT
            ttft_results = self.measure_ttft(quantized_model)
            
            # Measure memory
            memory = self.measure_memory(quantized_model)
            
            results[quant] = {
                'size_gb': self._get_model_size(quant),
                'ram_gb': memory['ram_gb'],
                'tps_output': tps_results['500_tokens']['mean_tps'],  # Use 500 tokens as standard
                'tps_std': tps_results['500_tokens']['std_tps'],
                'ttft_ms': ttft_results['2048_context']['mean_ttft_ms'],  # Use 2048 context as standard
                'ttft_std': ttft_results['2048_context']['std_ttft_ms']
            }
        
        # Add disclaimer
        results['_disclaimer'] = (
            "⚠️ CPU-ONLY BENCHMARKS: All performance metrics measured on "
            f"{self.HARDWARE_SPEC['platform']} with {self.HARDWARE_SPEC['cpu_cores']} cores. "
            "GPU inference may be 5-20x faster depending on model and quantization. "
            "These metrics are for RELATIVE comparison only."
        )
        
        return results
```

#### 2.2 Integrate into Main Benchmark

**File:** `benchmarks/evaluation/run_benchmark.py`

```python
def run_cpu_performance_benchmarks(self, model) -> Dict:
    """Run CPU-only performance benchmarks"""
    from cpu_performance import CPUPerformanceBenchmark
    
    perf_bench = CPUPerformanceBenchmark(model, self.config)
    return perf_bench.run_full_benchmark(model)
```

### Phase 3: Update Registry Schema

#### 3.1 Enhance Registry Update Script

**File:** `scripts/update_registry.py`

```python
def update_registry(submission_file, results_file, registry_file):
    # ... existing code ...
    
    # Build performance metrics (enhanced)
    performance = {
        'hardware': 'GitHub Actions (2-core CPU)',
        'quantizations': {}
    }
    
    # Extract CPU performance results
    cpu_perf = results.get('cpu_performance', {})
    if cpu_perf:
        for quant, metrics in cpu_perf.items():
            if quant == '_disclaimer':
                continue
            performance['quantizations'][quant] = {
                'size_gb': metrics.get('size_gb'),
                'ram_gb': metrics.get('ram_gb'),
                'tps_output': metrics.get('tps_output'),
                'ttft_ms': metrics.get('ttft_ms')
            }
    
    # Build domain scores
    domain_scores = {
        'general': aggregate_score,
        'finance': results.get('domain_scores', {}).get('finance'),
        'healthcare': results.get('domain_scores', {}).get('healthcare'),
        'legal': results.get('domain_scores', {}).get('legal'),
        'coding': use_cases.get('coding', {}).get('score')
    }
    
    # Determine deployment targets
    deployment_targets = []
    min_ram = min([q.get('ram_gb', 999) for q in performance['quantizations'].values()], default=999)
    
    if min_ram <= 2:
        deployment_targets.append('raspberry_pi_5')
    if min_ram <= 4:
        deployment_targets.append('jetson_orin')
    if min_ram <= 8:
        deployment_targets.append('16gb_laptop')
    if min_ram <= 16:
        deployment_targets.append('mobile_high_end')
    
    # Create model entry (enhanced)
    model_entry = {
        # ... existing fields ...
        'performance': performance,
        'domain_scores': domain_scores,
        'deployment_targets': deployment_targets,
        'compliance': {
            'hipaa': False,  # TODO: Add compliance checking
            'gdpr': True,     # Default to True for open models
            'finra': False
        },
        'tags': model.get('tags', []),
        'best_for': best_for,
        'not_for': not_for
    }
```

### Phase 4: Website Enhancements

#### 4.1 Model Detail Page

**File:** `website/model.html`

**Add:**
- Use case performance tabs
- Performance charts (TPS, TTFT, Memory)
- Deployment guide
- Benchmark deep-dive

#### 4.2 Marketplace Filtering

**File:** `website/assets/js/marketplace.js`

**Enhance:**
- RAM filter
- Context length filter
- Use case-specific sorting
- Performance-based filtering (TPS, TTFT)

---

## How to Submit a Model (Complete Guide)

### Step 1: Fork Repository

```bash
git clone https://github.com/2796gaurav/slm-benchmark.git
cd slm-benchmark
```

### Step 2: Create Submission File

```bash
cp models/submissions/template.yml models/submissions/my-model.yml
```

Edit `my-model.yml`:

```yaml
model:
  name: "MyModel-1.5B"
  family: "MyFamily"
  hf_repo: "username/MyModel-1.5B"
  parameters: "1.5B"  # Must be ≤3B
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

### Step 3: Validate Locally (Optional)

```bash
python .github/scripts/validate_submission.py \
  --submission-file models/submissions/my-model.yml
```

### Step 4: Create Pull Request

```bash
git checkout -b submit-my-model
git add models/submissions/my-model.yml
git commit -m "Submit MyModel-1.5B"
git push origin submit-my-model
```

Then open PR on GitHub.

### Step 5: Wait for Auto-Validation

The workflow will:
1. Auto-detect model properties
2. Post results as PR comment
3. Validate submission

### Step 6: Maintainer Runs Tests

Maintainer will comment:
- `/quick-test` - Fast smoke test (~5 min)
- `/full-benchmark` - Complete evaluation (~2 hours)
- `/publish-results` - Add to marketplace

### Step 7: Model Published

Once `/publish-results` is run:
- Model added to `models/registry.json`
- Website updated
- PR merged
- Model appears on marketplace

---

## Key Code Locations

### Submission Flow
- **Template:** `models/submissions/template.yml`
- **Validation:** `.github/scripts/validate_submission.py`
- **Auto-detection:** `benchmarks/validation/auto_detector.py`
- **Workflow:** `.github/workflows/02-submission-workflow.yml`

### Benchmarking
- **Main runner:** `benchmarks/evaluation/run_benchmark.py`
- **Edge metrics:** `benchmarks/evaluation/edge_benchmark.py`
- **Safety:** `benchmarks/evaluation/safety_eval.py`
- **Long context:** `benchmarks/evaluation/long_context_eval.py`

### Registry & Website
- **Registry update:** `scripts/update_registry.py`
- **Report generation:** `scripts/generate_report.py`
- **Website sync:** `scripts/update_website.py`
- **Registry:** `models/registry.json`
- **Website data:** `website/assets/data/leaderboard.json`

### Frontend
- **Homepage:** `website/index.html`
- **Model page:** `website/model.html`
- **Marketplace JS:** `website/assets/js/marketplace.js`
- **Leaderboard JS:** `website/assets/js/leaderboard.js`

---

## Conclusion

The codebase has a **solid foundation** with:
- ✅ Complete submission workflow
- ✅ Core benchmarking infrastructure
- ✅ Registry system
- ✅ Website frontend

**Missing pieces** for full marketplace vision:
- ❌ RAG/Function Calling specific benchmarks
- ❌ CPU performance benchmarking (TPS/TTFT)
- ❌ Domain-specific benchmarks
- ❌ Enhanced model detail pages

The architecture is **well-designed** and **extensible**. Adding the missing components follows the existing patterns and should be straightforward.
