#!/usr/bin/env python3
"""Edge device performance benchmarks"""

import time
import psutil
import torch
from typing import Dict

class EdgeBenchmark:
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def measure_latency(self) -> Dict:
        """Measure inference latency"""
        prompt = "Write a short story about"
        tokens = []
        
        for _ in range(10):  # Warmup
            _ = self.model.generate(prompt, max_tokens=50)
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = self.model.generate(prompt, max_tokens=50)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            'mean_ms': sum(latencies) / len(latencies),
            'p50_ms': sorted(latencies)[50],
            'p95_ms': sorted(latencies)[95],
            'p99_ms': sorted(latencies)[99]
        }
    
    def measure_throughput(self) -> Dict:
        """Measure tokens per second"""
        prompt = "Write a short story about"
        
        start = time.perf_counter()
        total_tokens = 0
        
        for _ in range(10):
            output = self.model.generate(prompt, max_tokens=100)
            total_tokens += 100
        
        elapsed = time.perf_counter() - start
        
        return {
            'tokens_per_second': total_tokens / elapsed,
            'seconds_per_token': elapsed / total_tokens
        }
    
    def measure_memory(self) -> Dict:
        """Measure memory usage"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = self.model.generate("Test", max_tokens=100)
            
            return {
                'gpu_memory_mb': torch.cuda.max_memory_allocated() / 1e6,
                'gpu_memory_reserved_mb': torch.cuda.max_memory_reserved() / 1e6
            }
        else:
            process = psutil.Process()
            return {
                'cpu_memory_mb': process.memory_info().rss / 1e6
            }
    
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