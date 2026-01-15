#!/usr/bin/env python3
"""
CPU-only performance benchmarking for GitHub Actions runners.
All metrics are hardware-specific and NOT used for ranking.
"""

import time
import psutil
import numpy as np
import torch
import logging
from typing import Dict, List, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class CPUPerformanceBenchmark:
    """
    CPU-only performance benchmarking system.
    
    Hardware Specification:
    - Platform: GitHub Actions Standard Runner
    - CPU: 2-core x86_64 @ 2.60GHz
    - RAM: 7 GB
    - Storage: 14 GB SSD
    
    All metrics are CPU-specific and may not reflect GPU performance.
    """
    
    HARDWARE_SPEC = {
        'platform': 'GitHub Actions Standard Runner',
        'cpu_cores': 2,
        'cpu_model': 'x86_64 @ 2.60GHz',
        'ram_gb': 7,
        'storage_gb': 14
    }
    
    def __init__(self, model, tokenizer, config):
        """
        Args:
            model: Loaded model instance (HFLM or transformers)
            tokenizer: Tokenizer instance
            config: BenchmarkConfig
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._check_model_interface()
    
    def _check_model_interface(self):
        """Determine model interface type"""
        # Check if it's an HFLM wrapper
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'generate'):
            self.model_obj = self.model.model
            self.is_hflm = True
        elif hasattr(self.model, 'generate'):
            self.model_obj = self.model
            self.is_hflm = False
        else:
            raise ValueError("Model must have generate() method")
    
    def measure_tps(self, output_lengths: List[int] = [100, 500, 1000], num_runs: int = 5) -> Dict:
        """
        Measure Tokens Per Second (OUTPUT GENERATION ONLY).
        
        This is the most compute-intensive metric and measures
        the model's generation speed, not prefill.
        
        Args:
            output_lengths: List of output token lengths to test
            num_runs: Number of runs per length
        
        Returns:
            Dict with TPS metrics per output length
        """
        logger.info("Measuring TPS (Tokens Per Second)...")
        results = {}
        
        prompt = "The meaning of life is"
        
        for output_len in output_lengths:
            logger.info(f"  Testing {output_len} token output...")
            tps_values = []
            
            for run in range(num_runs):
                try:
                    # Warm-up run (first iteration only)
                    if run == 0:
                        _ = self._generate(prompt, max_new_tokens=10)
                    
                    # Tokenize input
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Measure generation time
                    start = time.perf_counter()
                    outputs = self._generate(
                        prompt,
                        max_new_tokens=output_len,
                        do_sample=False,  # Deterministic
                    )
                    end = time.perf_counter()
                    
                    # Calculate TPS
                    elapsed = end - start
                    
                    # Count generated tokens
                    if isinstance(outputs, torch.Tensor):
                        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
                    elif isinstance(outputs, (list, tuple)):
                        tokens_generated = len(outputs) - len(inputs['input_ids'][0])
                    else:
                        # Fallback: estimate from output length
                        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
                        tokens_generated = len(self.tokenizer.encode(output_text)) - len(inputs['input_ids'][0])
                    
                    if elapsed > 0:
                        tps = tokens_generated / elapsed
                        tps_values.append(tps)
                    
                except Exception as e:
                    logger.warning(f"  Run {run+1} failed: {e}")
                    continue
            
            if tps_values:
                results[f'{output_len}_tokens'] = {
                    'mean_tps': float(np.mean(tps_values)),
                    'std_tps': float(np.std(tps_values)),
                    'median_tps': float(np.median(tps_values)),
                    'min_tps': float(np.min(tps_values)),
                    'max_tps': float(np.max(tps_values)),
                    'samples': len(tps_values)
                }
            else:
                logger.warning(f"  No valid TPS measurements for {output_len} tokens")
                results[f'{output_len}_tokens'] = {
                    'mean_tps': 0.0,
                    'error': 'No valid measurements'
                }
        
        return results
    
    def measure_ttft(self, context_lengths: List[int] = [512, 2048, 8192], num_runs: int = 5) -> Dict:
        """
        Measure Time to First Token (Prefill Latency).
        
        Important for interactive applications where users
        expect immediate feedback.
        
        Args:
            context_lengths: List of context lengths to test
            num_runs: Number of runs per context length
        
        Returns:
            Dict with TTFT metrics per context length
        """
        logger.info("Measuring TTFT (Time to First Token)...")
        results = {}
        
        for ctx_len in context_lengths:
            logger.info(f"  Testing {ctx_len} token context...")
            ttft_values = []
            
            for run in range(num_runs):
                try:
                    # Generate context of specified length
                    # Use repeated words to create context
                    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
                    prompt = " ".join(words * (ctx_len // len(words) + 1))
                    prompt = prompt[:ctx_len * 5]  # Rough estimate: 5 chars per token
                    
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Measure time to first token
                    start = time.perf_counter()
                    _ = self._generate(
                        prompt,
                        max_new_tokens=1,  # Just first token
                        do_sample=False
                    )
                    end = time.perf_counter()
                    
                    ttft_ms = (end - start) * 1000
                    ttft_values.append(ttft_ms)
                    
                except Exception as e:
                    logger.warning(f"  Run {run+1} failed: {e}")
                    continue
            
            if ttft_values:
                results[f'{ctx_len}_context'] = {
                    'mean_ttft_ms': float(np.mean(ttft_values)),
                    'std_ttft_ms': float(np.std(ttft_values)),
                    'median_ttft_ms': float(np.median(ttft_values)),
                    'min_ttft_ms': float(np.min(ttft_values)),
                    'max_ttft_ms': float(np.max(ttft_values)),
                    'samples': len(ttft_values)
                }
            else:
                logger.warning(f"  No valid TTFT measurements for {ctx_len} context")
                results[f'{ctx_len}_context'] = {
                    'mean_ttft_ms': 0.0,
                    'error': 'No valid measurements'
                }
        
        return results
    
    def measure_memory(self) -> Dict:
        """
        Measure RAM usage during inference.
        
        Returns:
            Dict with memory metrics
        """
        logger.info("Measuring Memory Usage...")
        
        try:
            process = psutil.Process()
            
            # Baseline memory (before model operations)
            baseline_mb = process.memory_info().rss / 1024 / 1024
            
            # Memory after model is loaded (should already be loaded)
            mem_loaded_mb = process.memory_info().rss / 1024 / 1024
            
            # Peak memory during generation
            prompt = "Tell me about artificial intelligence and machine learning"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            mem_before = process.memory_info().rss / 1024 / 1024
            _ = self._generate(prompt, max_new_tokens=100)
            mem_after = process.memory_info().rss / 1024 / 1024
            
            # Get peak memory
            peak_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                'baseline_mb': float(baseline_mb),
                'model_loaded_mb': float(mem_loaded_mb),
                'peak_generation_mb': float(mem_after),
                'overhead_mb': float(mem_after - baseline_mb),
                'ram_gb': float(mem_after / 1024)  # Convert to GB
            }
            
        except Exception as e:
            logger.error(f"Error measuring memory: {e}")
            return {
                'error': str(e),
                'ram_gb': 0.0
            }
    
    def _generate(self, prompt: str, max_new_tokens: int, **kwargs) -> torch.Tensor:
        """Unified generation interface"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get('do_sample', False),
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                **{k: v for k, v in kwargs.items() if k != 'do_sample'}
            )
        
        return outputs
    
    def get_hardware_spec(self) -> Dict:
        """Get actual hardware specifications"""
        try:
            return {
                'platform': self.HARDWARE_SPEC['platform'],
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'cpu_freq_ghz': psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else None,
                'ram_total_gb': psutil.virtual_memory().total / 1024**3,
                'ram_available_gb': psutil.virtual_memory().available / 1024**3
            }
        except Exception as e:
            logger.warning(f"Could not get hardware spec: {e}")
            return self.HARDWARE_SPEC
    
    def run_full_benchmark(self) -> Dict:
        """
        Run complete CPU performance benchmark.
        
        Returns:
            Dict with all performance metrics
        """
        logger.info("🚀 Starting CPU Performance Benchmark")
        logger.info(f"Hardware: {self.get_hardware_spec()}")
        
        # Measure TPS
        tps_results = self.measure_tps()
        
        # Measure TTFT
        ttft_results = self.measure_ttft()
        
        # Measure memory
        memory_results = self.measure_memory()
        
        # Compile results
        results = {
            'hardware': self.get_hardware_spec(),
            'tps': tps_results,
            'ttft': ttft_results,
            'memory': memory_results,
            'disclaimer': (
                "⚠️ CPU-ONLY BENCHMARKS: Measured on GitHub Actions (2-core CPU). "
                "GPU inference may be 5-20x faster. Use for RELATIVE comparison only."
            )
        }
        
        logger.info("✅ CPU Performance Benchmark Complete")
        return results
