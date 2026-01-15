import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate readable report from benchmark results")
    parser.add_argument("--results-dir", required=True, help="Directory containing raw results")
    parser.add_argument("--output", required=True, help="Path to save the processed JSON report")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Scanning results in {results_dir}")
    
    # Aggregate results from found files
    found_files = list(results_dir.glob("**/*.json"))
    
    # Initialize aggregation structure with all marketplace benchmarks
    full_report = {
        "aggregate_scores": {},
        "detailed": {},
        "aggregate_score": 0.0,
        "reasoning_scores": {},
        "coding_scores": {},
        "math_scores": {},
        "language_scores": {},
        "edge_metrics": {},
        "safety_scores": {},
        "rag_scores": {},
        "function_calling_scores": {},
        "guardrails_scores": {},
        "domain_scores": {},
        "cpu_performance": {}
    }
    
    valid_results = 0
    total_aggregate = 0.0
    
    for f in found_files:
        if f.name == "summary.json":
            continue
            
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
                
                # Check if this is a valid benchmark result
                if 'aggregate_score' in data:
                    full_report["detailed"][f.name] = data
                    total_aggregate += data['aggregate_score']
                    valid_results += 1
                    
                    # Merge all benchmark category scores
                    benchmark_keys = [
                        "reasoning_scores", "coding_scores", "math_scores", "language_scores",
                        "edge_metrics", "safety_scores", "rag_scores", "function_calling_scores",
                        "guardrails_scores", "domain_scores", "cpu_performance"
                    ]
                    for key in benchmark_keys:
                        if key in data and data[key]:
                            # Merge or update existing data
                            if key not in full_report or not full_report[key]:
                                full_report[key] = data[key]
                            elif isinstance(full_report[key], dict) and isinstance(data[key], dict):
                                # Merge dictionaries
                                full_report[key].update(data[key])
                            
        except Exception as e:
            print(f"Skipping invalid file {f}: {e}")
    
    # Calculate final average aggregate score across quantizations
    if valid_results > 0:
        full_report["aggregate_score"] = total_aggregate / valid_results
    else:
        print("Warning: No valid benchmark results found.")

    # Save processed report
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"Report generated at {args.output}")

if __name__ == "__main__":
    main()
