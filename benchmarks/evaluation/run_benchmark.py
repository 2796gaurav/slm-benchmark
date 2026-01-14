#!/usr/bin/env python3
"""
SLM Benchmark - Main Benchmark Runner
Production-ready, deterministic, and comprehensive evaluation
"""

import os
import sys
import json
import yaml
import time
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Import evaluation frameworks
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import custom benchmarks
from edge_benchmark import EdgeBenchmark
from quantization_bench import QuantizationBenchmark
from safety_eval import SafetyEvaluator
from carbon_tracker import CarbonTrackerWrapper
from fine_tuning_benchmark import FineTuningEfficiency
from long_context_eval import LongContextEvaluator
from bias_fairness_eval import BiasAndFairnessEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run.
    
    Args:
        model_name: Display name of the model
        hf_repo: HuggingFace repository ID (e.g., 'HuggingFaceTB/SmolLM2-1.7B')
        parameters: Model size (e.g., '1.7B')
        quantization: Quantization format ('FP16', 'Q4_K_M', 'Q8_0')
        tasks: List of task categories to evaluate
        num_fewshot: Number of few-shot examples (default: 5)
        batch_size: Batch size for inference (default: 8)
        seed: Random seed for reproducibility (default: 42)
        deterministic: Enable deterministic mode for reproducibility (default: True)
        edge_tests: Run edge performance benchmarks (default: False)
        safety_tests: Run safety and bias tests (default: True)
        enable_carbon_tracking: Track energy consumption (default: False)
        limit: Limit samples per task for testing (default: None = no limit)
    """
    model_name: str
    hf_repo: str
    parameters: str
    quantization: str
    tasks: List[str]
    num_fewshot: int = 5
    batch_size: int = 8
    seed: int = 42
    deterministic: bool = True
    edge_tests: bool = False
    safety_tests: bool = True
    enable_carbon_tracking: bool = False
    limit: int = None  # Limit number of samples per task


@dataclass
class BenchmarkResult:
    """Results from benchmark run"""
    model_name: str
    quantization: str
    timestamp: str
    system_info: Dict
    
    # Task results
    reasoning_scores: Dict
    coding_scores: Dict
    math_scores: Dict
    language_scores: Dict
    
    # Edge performance
    edge_metrics: Dict
    
    # Safety
    safety_scores: Dict
    
    # Aggregate
    aggregate_score: float
    rank: int
    
    # New Categories
    long_context_scores: Dict = None
    fine_tuning_metrics: Dict = None
    
    # Environmental
    environmental_metrics: Dict = None
    efficiency_score: float = 0.0


class SLMBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
        # Set deterministic mode
        if config.deterministic:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(config.seed)
        
        # Initialize system info
        self.system_info = self._get_system_info()

        # Initialize carbon tracker
        self.carbon_tracker = CarbonTrackerWrapper(
            output_dir="results/carbon",
            enable=config.enable_carbon_tracking
        )
    
    def _get_system_info(self) -> Dict:
        """Get system information for reproducibility"""
        import platform
        import subprocess
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info


    def run_long_context_benchmarks(self, model) -> Dict:
        """Run long context benchmarks using LongContextEvaluator.

        These metrics are reported for transparency but are not currently
        included in the aggregate ranking score.
        """
        logger.info("Running long context benchmarks...")
        evaluator = LongContextEvaluator(model, self.config)
        return evaluator.evaluate_all()

    def run_fine_tuning_benchmarks(self, model_name) -> Dict:
        """Run fine-tuning efficiency benchmarks.

        NOTE: This implementation is heuristic and does not perform real
        fine-tuning (to remain CPU- and CI-friendly). The metrics are
        reported for reference only and are not included in rankings.
        """
        logger.info("Running fine-tuning benchmarks (heuristic only)...")
        ft_bench = FineTuningEfficiency()
        return ft_bench.evaluate_tunability(model_name)
    
    def load_model(self):
        """Load model from Hugging Face.
        
        Returns:
            Loaded model instance (HFLM or Llama)
            
        Raises:
            ValueError: If quantization format is unsupported
            Exception: If model loading fails
        """
        logger.info(f"Loading model: {self.config.hf_repo}")
        logger.info(f"Quantization: {self.config.quantization}")
        logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        try:
            # Load with specific quantization if specified
            # NOTE: To keep the benchmark portable and GitHub Actions‑friendly,
            # we always evaluate via the Hugging Face backend on CPU/GPU.
            # Quantization metadata is still recorded but does not change
            # the evaluation backend in this runner.
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = 'float16' if self.config.quantization.upper().startswith('FP16') and torch.cuda.is_available() else None

            hf_kwargs = {
                "pretrained": self.config.hf_repo,
                "device": device,
                "trust_remote_code": True,
            }
            if dtype is not None:
                hf_kwargs["dtype"] = dtype

            logger.info(f"Attempting to load model with kwargs: {hf_kwargs}")
            model = HFLM(**hf_kwargs)
            
            logger.info("Model loaded successfully")
            return model
            
        except ValueError as e:
            logger.error(f"Unsupported quantization format: {self.config.quantization}")
            logger.error("Supported formats: FP16, FP32, transformers")
            raise ValueError(f"Unsupported quantization: {self.config.quantization}") from e
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Try: pip install transformers accelerate")
            raise RuntimeError(f"Dependency error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Model repo: {self.config.hf_repo}")
            logger.error("Common fixes:")
            logger.error("  1. Verify model exists on HuggingFace Hub")
            logger.error("  2. Check if model requires trust_remote_code=True")
            logger.error("  3. Ensure you have internet connection")
            logger.error("  4. Try: huggingface-cli login (if private model)")
            raise RuntimeError(f"Model loading failed for {self.config.hf_repo}: {e}") from e
    
    def run_reasoning_benchmarks(self, model) -> Dict:
        """Run reasoning benchmarks"""
        logger.info("Running reasoning benchmarks...")
        
        tasks = [
            'mmlu',
            'arc_challenge',
            'hellaswag',
            'truthfulqa_mc2'
        ]
        
        try:
            results = evaluator.simple_evaluate(
                model=model,
                tasks=tasks,
                num_fewshot=self.config.num_fewshot,
                batch_size=self.config.batch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                limit=self.config.limit
            )
            logger.info("Reasoning benchmarks completed")
            return results['results']
        except Exception as e:
            logger.error(f"Reasoning benchmarks failed: {e}")
            logger.warning("Returning empty results for reasoning")
            return {}
    
    def run_coding_benchmarks(self, model) -> Dict:
        """Run coding benchmarks"""
        logger.info("Running coding benchmarks...")
        
        tasks = ['humaneval', 'mbpp']
        
        try:
            results = evaluator.simple_evaluate(
                model=model,
                tasks=tasks,
                num_fewshot=0, # Coding tasks usually differ in fewshot support, safer to use 0
                batch_size=1,  # Coding tasks need batch_size=1
                device='cuda' if torch.cuda.is_available() else 'cpu',
                limit=self.config.limit,
                confirm_run_unsafe_code=True  # Required for executing coding benchmarks
            )
            return results['results']
        except Exception as e:
            logger.error(f"Coding benchmarks failed: {e}")
            return {}
    
    def run_math_benchmarks(self, model) -> Dict:
        """Run math benchmarks"""
        logger.info("Running math benchmarks...")
        
        tasks = ['gsm8k', 'math_qa']
        
        try:
            results = evaluator.simple_evaluate(
                model=model,
                tasks=tasks,
                num_fewshot=self.config.num_fewshot,
                batch_size=self.config.batch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                limit=self.config.limit
            )
            return results['results']
        except Exception as e:
            logger.error(f"Math benchmarks failed: {e}")
            return {}
    
    def run_language_benchmarks(self, model) -> Dict:
        """Run language understanding benchmarks"""
        logger.info("Running language benchmarks...")
        
        tasks = [
            'boolq',
            'piqa',
            'winogrande'
        ]
        
        try:
            results = evaluator.simple_evaluate(
                model=model,
                tasks=tasks,
                num_fewshot=self.config.num_fewshot,
                batch_size=self.config.batch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                limit=self.config.limit
            )
            return results['results']
        except Exception as e:
            logger.error(f"Math benchmarks failed: {e}")
            return {}
    
    def run_edge_benchmarks(self, model) -> Dict:
        """Run edge performance tests"""
        if not self.config.edge_tests:
            logger.info("Edge benchmarks disabled (hardware-dependent metrics).")
            return {}

        logger.info("Running edge benchmarks (hardware-dependent, excluded from ranking)...")

        edge_bench = EdgeBenchmark(model, self.config)

        metrics = {
            'latency': edge_bench.measure_latency(),
            'throughput': edge_bench.measure_throughput(),
            'memory_usage': edge_bench.measure_memory(),
            'energy_efficiency': edge_bench.measure_energy()
        }

        return metrics
    
    def run_safety_benchmarks(self, model) -> Dict:
        """Run safety and bias tests"""
        if not self.config.safety_tests:
            return {}

        logger.info("Running safety benchmarks...")

        safety_eval = SafetyEvaluator(model, self.config)
        bias_fairness_eval = BiasAndFairnessEvaluator(model, self.config)

        scores = {
            'toxicity': safety_eval.measure_toxicity(),
            'bias': safety_eval.measure_bias(),
            'truthfulness': safety_eval.measure_truthfulness(),
            # Bias & fairness is reported as a nested structure; it is not
            # folded into the scalar safety score used for ranking.
            'bias_fairness': bias_fairness_eval.evaluate_all(),
        }

        return scores
    
    def calculate_aggregate_score(self) -> float:
        """Calculate weighted aggregate score"""
        # Hardware-dependent metrics (edge, efficiency, carbon) are *not*
        # included in the aggregate score used for rankings. This keeps
        # scores comparable across heterogeneous environments (e.g. local
        # runs vs. GitHub Actions CPU runners).
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
    
    def _average_scores(self, score_dict: Dict) -> float:
        """Calculate average of scores in dict"""
        if not score_dict:
            return 0.0
        
        values = []
        for v in score_dict.values():
            if isinstance(v, dict):
                # Handle lm-eval style metrics (acc,none, acc_norm,none, etc.)
                # Priority: acc_norm > acc > any key starting with acc
                acc_norm = v.get('acc_norm,none', v.get('acc_norm'))
                acc = v.get('acc,none', v.get('acc'))
                
                if acc_norm is not None:
                    values.append(float(acc_norm) * 100)
                elif acc is not None:
                    values.append(float(acc) * 100)
                else:
                    # Fallback: check for any key containing 'acc'
                    for k, val in v.items():
                        if 'acc' in k and isinstance(val, (int, float)):
                            values.append(float(val) * 100)
                            break
            elif isinstance(v, (int, float)):
                values.append(float(v))
        
        return sum(values) / len(values) if values else 0.0
    
    def run_full_benchmark(self) -> BenchmarkResult:
        """Run complete benchmark suite"""
        logger.info(f"Starting benchmark for {self.config.model_name}")
        start_time = time.time()
        
        # Start carbon tracking
        self.carbon_tracker.start()
        
        
        # Load model
        model = self.load_model()
        
        # Run all benchmarks
        self.results['reasoning_scores'] = self.run_reasoning_benchmarks(model)
        self.results['coding_scores'] = self.run_coding_benchmarks(model)
        self.results['math_scores'] = self.run_math_benchmarks(model)
        self.results['language_scores'] = self.run_language_benchmarks(model)
        self.results['edge_metrics'] = self.run_edge_benchmarks(model)
        self.results['safety_scores'] = self.run_safety_benchmarks(model)
        # self.results['tool_use_scores'] = self.run_tool_use_benchmarks(model) - Removed as per request
        self.results['long_context_scores'] = self.run_long_context_benchmarks(model)
        self.results['fine_tuning_metrics'] = self.run_fine_tuning_benchmarks(self.config.model_name)
        
        # Calculate aggregate
        aggregate_score = self.calculate_aggregate_score()
        
        # Stop carbon tracking
        env_metrics = self.carbon_tracker.stop()
        
        # Calculate efficiency score
        efficiency_score = 0.0
        if env_metrics and env_metrics.get('energy_consumed_kwh', 0) > 0:
            # Formula: Accuracy / Energy(kWh)
            # Baseline: 1 kWh
            efficiency_score = (aggregate_score / env_metrics['energy_consumed_kwh'])
        elif env_metrics: 
             # Fallback if energy is too low to measure (very fast run)
             # Avoid infinity, set to a high cap based on latency?
             # For now, just set to 0 or handle effectively.
             efficiency_score = 0.0 # Or maybe aggregate_score * 10 (arbitrary boost for efficiency)
             if aggregate_score > 0:
                  logger.warning("Energy consumption near zero, efficiency score may be inaccurate")
        
        
        # Create result object
        result = BenchmarkResult(
            model_name=self.config.model_name,
            quantization=self.config.quantization,
            timestamp=datetime.now().isoformat(),
            system_info=self.system_info,
            reasoning_scores=self.results['reasoning_scores'],
            coding_scores=self.results['coding_scores'],
            math_scores=self.results['math_scores'],
            language_scores=self.results['language_scores'],
            edge_metrics=self.results['edge_metrics'],
            safety_scores=self.results['safety_scores'],
            long_context_scores=self.results['long_context_scores'],
            fine_tuning_metrics=self.results['fine_tuning_metrics'],
            aggregate_score=aggregate_score,
            environmental_metrics=env_metrics,
            efficiency_score=efficiency_score,
            rank=0  # Will be calculated later
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Benchmark completed in {elapsed:.2f} seconds")
        logger.info(f"Aggregate score: {aggregate_score:.2f}")
        
        return result
    
    def save_results(self, result: BenchmarkResult, output_dir: Path):
        """Save results to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        result_dict = asdict(result)
        safe_model_name = self.config.model_name.replace('/', '_')
        output_file = output_dir / f"{safe_model_name}_{self.config.quantization}_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Run SLM Benchmark')
    parser.add_argument('--submission-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/raw/')
    parser.add_argument('--save-logs', action='store_true')
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--full-report', action='store_true')
    parser.add_argument('--enable-carbon-tracking', action='store_true', help='Enable energy consumption tracking')
    parser.add_argument('--limit', type=int, help='Limit number of samples per task for testing')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    
    args = parser.parse_args()
    
    if not args.timestamp:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load submission
    with open(args.submission_file, 'r') as f:
        submission = yaml.safe_load(f)
    
    model_info = submission['model']
    
    
    # Run benchmarks for each quantization
    all_results = []
    quantizations = model_info.get('quantizations', [{'name': 'FP16'}])
    if not quantizations:
        quantizations = [{'name': 'FP16'}]

    for quant_info in quantizations:
        config = BenchmarkConfig(
            model_name=model_info.get('hf_repo', model_info.get('name', args.submission_file)),
            hf_repo=model_info.get('hf_repo'),
            parameters=model_info.get('parameters', {}),
            quantization=quant_info['name'],
            tasks=model_info.get('categories', []),
            seed=args.seed,
            deterministic=args.deterministic,
            limit=args.limit,
            enable_carbon_tracking=args.enable_carbon_tracking,
            batch_size=args.batch_size
        )
        
        benchmark = SLMBenchmark(config)
        
        try:
            result = benchmark.run_full_benchmark()
            
            # Save results
            output_dir = Path(args.output_dir) / model_info['name'] / args.timestamp
            result_file = benchmark.save_results(result, output_dir)
            
            all_results.append({
                'quantization': quant_info['name'],
                'result_file': str(result_file),
                'aggregate_score': result.aggregate_score,
                'efficiency_score': result.efficiency_score,
                'environmental_metrics': result.environmental_metrics
            })
            
        except Exception as e:
            logger.error(f"Benchmark failed for {quant_info['name']}: {e}")
            continue
    
    # Save summary
    summary = {
        'model': model_info['name'],
        'timestamp': args.timestamp,
        'results': all_results,
        'status': 'completed' if all_results else 'failed'
    }
    
    summary_file = Path(args.output_dir) / model_info['name'] / args.timestamp / 'summary.json'
    # Ensure directory exists even if benchmark failed
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Benchmark suite completed. Summary: {summary_file}")


if __name__ == '__main__':
    main()