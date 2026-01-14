#!/usr/bin/env python3
"""Edge device performance benchmarks"""

import time
import psutil
import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class EdgeBenchmark:
    """Benchmark edge device performance metrics."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self._check_model_interface()
    
    
    def _check_model_interface(self):
        """Check which model interface we're using."""
        self.is_hf_model = hasattr(self.model, 'generate') and hasattr(self.model, 'model')
        self.is_llama_cpp = hasattr(self.model, '__call__') and not self.is_hf_model
        logger.info(f"Model interface: HF={self.is_hf_model}, llama.cpp={self.is_llama_cpp}")
    
    def _generate_text(self, prompt: str, max_tokens: int = 50) -> Optional[str]:
        """Generate text using appropriate model interface."""
        try:
            if self.is_hf_model:
                # HuggingFace model
                output = self.model.generate(prompt, max_new_tokens=max_tokens)
                return output
            elif self.is_llama_cpp:
                # llama.cpp model
                output = self.model(prompt, max_tokens=max_tokens)
                if isinstance(output, dict) and 'choices' in output:
                    return output['choices'][0]['text']
                return str(output)
            else:
                # Generic interface
                return self.model.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None
    
    def measure_latency(self) -> Dict:
        """Measure inference latency.
        
        Returns:
            Dictionary with latency statistics (mean, p50, p95, p99 in milliseconds)
        """
        prompt = "Write a short story about"
        
        try:
            # Warmup
            for _ in range(10):
                output = self._generate_text(prompt, max_tokens=50)
                if output is None:
                    logger.warning("Warmup generation failed")
                    break
            
            # Measure latencies
            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                output = self._generate_text(prompt, max_tokens=50)
                if output is not None:
                    latencies.append((time.perf_counter() - start) * 1000)
            
            if not latencies:
                logger.error("No successful latency measurements")
                return {'error': 'Failed to measure latency'}
            
            latencies.sort()
            n = len(latencies)
            
            return {
                'mean_ms': sum(latencies) / n,
                'p50_ms': latencies[n // 2],
                'p95_ms': latencies[int(n * 0.95)],
                'p99_ms': latencies[int(n * 0.99)],
                'samples': n
            }
        except Exception as e:
            logger.error(f"Error measuring latency: {e}")
            return {'error': str(e)}
    
    
    def measure_throughput(self) -> Dict:
        """Measure tokens per second.
        
        Returns:
            Dictionary with throughput metrics
        """
        prompt = "Write a short story about"
        
        try:
            start = time.perf_counter()
            total_tokens = 0
            successful_runs = 0
            
            for _ in range(10):
                output = self._generate_text(prompt, max_tokens=100)
                if output is not None:
                    total_tokens += 100  # Approximate
                    successful_runs += 1
            
            elapsed = time.perf_counter() - start
            
            if total_tokens == 0 or elapsed == 0:
                return {'error': 'No successful throughput measurements'}
            
            return {
                'tokens_per_second': total_tokens / elapsed,
                'seconds_per_token': elapsed / total_tokens,
                'successful_runs': successful_runs
            }
        except Exception as e:
            logger.error(f"Error measuring throughput: {e}")
            return {'error': str(e)}
    
    
    def measure_memory(self) -> Dict:
        """Measure memory usage.
        
        Returns:
            Dictionary with memory usage metrics
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                output = self._generate_text("Test", max_tokens=100)
                
                return {
                    'gpu_memory_mb': torch.cuda.max_memory_allocated() / 1e6,
                    'gpu_memory_reserved_mb': torch.cuda.max_memory_reserved() / 1e6
                }
            else:
                process = psutil.Process()
                mem_before = process.memory_info().rss
                output = self._generate_text("Test", max_tokens=100)
                mem_after = process.memory_info().rss
                
                return {
                    'cpu_memory_mb': mem_after / 1e6,
                    'memory_increase_mb': (mem_after - mem_before) / 1e6
                }
        except Exception as e:
            logger.error(f"Error measuring memory: {e}")
            return {'error': str(e)}
    
    def measure_energy(self) -> Dict:
        """Estimate energy efficiency"""
        # Simplified energy estimation
        throughput = self.measure_throughput()
        
        # Assume 250W TDP for GPU
        estimated_power_w = 250
        tokens_per_wh = throughput['tokens_per_second'] * 3600 / estimated_power_w
        
        return {
            'tokens_per_watt_hour': tokens_per_wh,
            'estimated_power_w': estimated_power_w
        }