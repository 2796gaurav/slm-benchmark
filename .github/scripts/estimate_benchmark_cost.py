import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

def estimate_benchmark_cost(model_parameters: str, num_quantizations: int) -> Dict:
    """
    Estimate GPU time and cost for benchmark run.
    Helps prevent accidental expensive runs.
    """
    
    # Parse parameters
    param_str = str(model_parameters).upper()
    if 'B' in param_str:
        params_b = float(param_str.replace('B', ''))
    elif 'M' in param_str:
        params_b = float(param_str.replace('M', '')) / 1000
    else:
        params_b = 1.0  # Default estimate
    
    # Estimate time per benchmark task (minutes)
    # Based on empirical data: ~2 min for 1B model, scales roughly linearly
    base_time_per_task = 2 * params_b
    
    # Number of benchmark tasks
    num_tasks = 15  # reasoning(4) + coding(2) + math(2) + language(3) + edge(2) + safety(2)
    
    # Total time per quantization
    time_per_quant = base_time_per_task * num_tasks
    
    # Total time for all quantizations
    total_time = time_per_quant * num_quantizations
    
    # Estimate cost (assuming $1/hour GPU time)
    estimated_cost = (total_time / 60) * 1.0
    
    return {
        'estimated_minutes': round(total_time, 1),
        'estimated_hours': round(total_time / 60, 2),
        'estimated_cost_usd': round(estimated_cost, 2),
        'num_tests': num_tasks,
        'num_quants': num_quantizations,
        'model_params': model_parameters
    }


def main_estimate_cost():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission-file', required=True)
    parser.add_argument('--output', default='cost_estimate.json')
    args = parser.parse_args()
    
    import yaml
    with open(args.submission_file, 'r') as f:
        submission = yaml.safe_load(f)
    
    model = submission['model']
    
    estimate = estimate_benchmark_cost(
        model['parameters'],
        len(model.get('quantizations', []))
    )
    
    with open(args.output, 'w') as f:
        json.dump(estimate, f, indent=2)
    
    print(f"Estimated benchmark time: {estimate['estimated_minutes']} minutes")
    print(f"Estimated cost: ${estimate['estimated_cost_usd']}")
    
    sys.exit(0)

if __name__ == "__main__":
    main_estimate_cost()
