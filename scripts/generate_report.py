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
    
    # Placeholder for aggregation logic
    # In a real scenario, we would parse specific metric files (MMLU, hellaswag, etc.)
    # Here we simulate finding results
    
    # Aggregate results from found files
    found_files = list(results_dir.glob("**/*.json"))
    
    # Initialize aggregation structure
    full_report = {
        "aggregate_scores": {},
        "detailed": {},
        "aggregate_score": 0.0,
        "reasoning_scores": {},
        "coding_scores": {},
        "math_scores": {},
        "language_scores": {},
        "edge_metrics": {},
        "safety_scores": {}
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
                    
                    # Merge individual category scores (simple first-found or last-found merge for now)
                    # Ideally we would average if multiple quantizations provided different scores
                    for key in ["reasoning_scores", "coding_scores", "math_scores", "language_scores", "edge_metrics", "safety_scores"]:
                        if key in data and data[key]:
                            full_report[key] = data[key]
                            
        except Exception as e:
            print(f"Skipping invalid file {f}: {e}")

    # Calculate final average aggregate score across quantizations
    if valid_results > 0:
        full_report["aggregate_score"] = total_aggregate / valid_results
    else:
        # Fallback for empty results
        print("Warning: No valid benchmark results found.")

    # Save processed report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"Report generated at {args.output}")

if __name__ == "__main__":
    main()
