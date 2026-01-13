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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    model_name: str
    hf_repo: str
    parameters: str
    quantization: str
    tasks: List[str]
    num_fewshot: int = 5
    batch_size: int = 8
    seed: int = 42
    deterministic: bool = True
    edge_tests: bool = True
    deterministic: bool = True
    edge_tests: bool = True
    safety_tests: bool = True
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
    
    def load_model(self):
        """Load model from Hugging Face"""
        logger.info(f"Loading model: {self.config.hf_repo}")
        
        try:
            # Load with specific quantization if specified
            if self.config.quantization == 'FP16':
                model = HFLM(
                    pretrained=self.config.hf_repo,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    dtype='float16'
                )
            elif 'Q4' in self.config.quantization or 'Q8' in self.config.quantization:
                # Load GGUF model
                from llama_cpp import Llama
                model_path = self._download_gguf_model()
                model = Llama(
                    model_path=model_path,
                    n_ctx=self.config.context_length,
                    n_gpu_layers=-1 if torch.cuda.is_available() else 0
                )
            else:
                model = HFLM(
                    pretrained=self.config.hf_repo,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def run_reasoning_benchmarks(self, model) -> Dict:
        """Run reasoning benchmarks"""
        logger.info("Running reasoning benchmarks...")
        
        tasks = [
            'mmlu',
            'arc_challenge',
            'hellaswag',
            'truthfulqa_mc2'
        ]
        
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=self.config.num_fewshot,
            batch_size=self.config.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            limit=self.config.limit
        )
        
        return results['results']
    
    def run_coding_benchmarks(self, model) -> Dict:
        """Run coding benchmarks"""
        logger.info("Running coding benchmarks...")
        
        tasks = ['humaneval', 'mbpp']
        
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=self.config.num_fewshot,
            batch_size=1,  # Coding tasks need batch_size=1
            device='cuda' if torch.cuda.is_available() else 'cpu',
            limit=self.config.limit,
            confirm_run_unsafe_code=True  # Required for executing coding benchmarks
        )
        
        return results['results']
    
    def run_math_benchmarks(self, model) -> Dict:
        """Run math benchmarks"""
        logger.info("Running math benchmarks...")
        
        tasks = ['gsm8k', 'math_qa']
        
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=self.config.num_fewshot,
            batch_size=self.config.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            limit=self.config.limit
        )
        
        return results['results']
    
    def run_language_benchmarks(self, model) -> Dict:
        """Run language understanding benchmarks"""
        logger.info("Running language benchmarks...")
        
        tasks = [
            'boolq',
            'piqa',
            'winogrande'
        ]
        
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=self.config.num_fewshot,
            batch_size=self.config.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            limit=self.config.limit
        )
        
        return results['results']
    
    def run_edge_benchmarks(self, model) -> Dict:
        """Run edge performance tests"""
        if not self.config.edge_tests:
            return {}
        
        logger.info("Running edge benchmarks...")
        
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
        
        scores = {
            'toxicity': safety_eval.measure_toxicity(),
            'bias': safety_eval.measure_bias(),
            'truthfulness': safety_eval.measure_truthfulness()
        }
        
        return scores
    
    def calculate_aggregate_score(self) -> float:
        """Calculate weighted aggregate score"""
        weights = {
            'reasoning': 0.30,
            'coding': 0.20,
            'math': 0.15,
            'language': 0.15,
            'edge': 0.10,
            'safety': 0.10
        }
        
        scores = {
            'reasoning': self._average_scores(self.results['reasoning_scores']),
            'coding': self._average_scores(self.results['coding_scores']),
            'math': self._average_scores(self.results['math_scores']),
            'language': self._average_scores(self.results['language_scores']),
            'edge': self._average_scores(self.results['edge_metrics']),
            'safety': self._average_scores(self.results['safety_scores'])
        }
        
        aggregate = sum(scores[k] * weights[k] for k in weights.keys())
        return aggregate
    
    def _average_scores(self, score_dict: Dict) -> float:
        """Calculate average of scores in dict"""
        if not score_dict:
            return 0.0
        
        values = []
        for v in score_dict.values():
            if isinstance(v, dict) and 'acc' in v:
                values.append(v['acc'] * 100)
            elif isinstance(v, (int, float)):
                values.append(float(v))
        
        return sum(values) / len(values) if values else 0.0
    
    def run_full_benchmark(self) -> BenchmarkResult:
        """Run complete benchmark suite"""
        logger.info(f"Starting benchmark for {self.config.model_name}")
        start_time = time.time()
        
        # Load model
        model = self.load_model()
        
        # Run all benchmarks
        self.results['reasoning_scores'] = self.run_reasoning_benchmarks(model)
        self.results['coding_scores'] = self.run_coding_benchmarks(model)
        self.results['math_scores'] = self.run_math_benchmarks(model)
        self.results['language_scores'] = self.run_language_benchmarks(model)
        self.results['edge_metrics'] = self.run_edge_benchmarks(model)
        self.results['safety_scores'] = self.run_safety_benchmarks(model)
        
        # Calculate aggregate
        aggregate_score = self.calculate_aggregate_score()
        
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
            aggregate_score=aggregate_score,
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
        output_file = output_dir / f"{self.config.model_name}_{self.config.quantization}_{int(time.time())}.json"
        
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
    parser.add_argument('--limit', type=int, help='Limit number of samples per task for testing')
    
    args = parser.parse_args()
    
    # Load submission
    with open(args.submission_file, 'r') as f:
        submission = yaml.safe_load(f)
    
    model_info = submission['model']
    
    # Run benchmark for each quantization
    all_results = []
    
    for quant in model_info['quantizations']:
        config = BenchmarkConfig(
            model_name=model_info['name'],
            hf_repo=model_info['hf_repo'],
            parameters=model_info['parameters'],
            quantization=quant['name'],
            tasks=model_info.get('categories', []),
            seed=args.seed,
            deterministic=args.deterministic,
            limit=args.limit
        )
        
        benchmark = SLMBenchmark(config)
        
        try:
            result = benchmark.run_full_benchmark()
            
            # Save results
            output_dir = Path(args.output_dir) / model_info['name'] / args.timestamp
            result_file = benchmark.save_results(result, output_dir)
            
            all_results.append({
                'quantization': quant['name'],
                'result_file': str(result_file),
                'aggregate_score': result.aggregate_score
            })
            
        except Exception as e:
            logger.error(f"Benchmark failed for {quant['name']}: {e}")
            continue
    
    # Save summary
    summary = {
        'model': model_info['name'],
        'timestamp': args.timestamp,
        'results': all_results,
        'status': 'completed' if all_results else 'failed'
    }
    
    summary_file = Path(args.output_dir) / model_info['name'] / args.timestamp / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Benchmark suite completed. Summary: {summary_file}")


if __name__ == '__main__':
    main()