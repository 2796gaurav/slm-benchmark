"""
Aggregates all benchmark results into a single JSON for the website
"""
import json
from pathlib import Path
from typing import List, Dict, Any

RESULTS_DIR = Path('results')
OUTPUT_FILE = Path('website/data/models.json')

def load_result(result_dir: Path) -> Dict[str, Any]:
    """Load a single model's results"""
    
    metadata_file = result_dir / 'metadata.json'
    accuracy_file = result_dir / 'accuracy.json'
    performance_file = result_dir / 'performance.json'
    
    if not all([metadata_file.exists(), accuracy_file.exists()]):
        print(f"⚠️  Incomplete results in {result_dir}")
        return None
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    with open(accuracy_file) as f:
        accuracy = json.load(f)
    
    performance = {}
    if performance_file.exists():
        with open(performance_file) as f:
            performance = json.load(f)
    
    # Extract key metrics
    model_data = {
        'name': metadata['model_config']['name'],
        'hf_model_id': metadata['model_config'].get('path', ''),
        'parameters': metadata['model_config']['size_params'],
        'quantization': metadata['model_config']['quantization'],
        'timestamp': metadata['timestamp'],
        'hardware': metadata['hardware']['device_name'],
        'accuracy': accuracy,
        'performance': performance,
        'avg_score': calculate_avg_score(accuracy)
    }
    
    return model_data

def calculate_avg_score(accuracy: Dict[str, float]) -> float:
    """Calculate average accuracy across all tasks"""
    if not accuracy:
        return 0.0
    
    # Weight important benchmarks
    weights = {
        'mmlu': 2.0,
        'arc_challenge': 1.5,
        'hellaswag': 1.5,
        'gsm8k': 2.0,
        'truthfulqa': 1.0,
        'winogrande': 1.0
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for task, score in accuracy.items():
        weight = weights.get(task, 1.0)
        weighted_sum += score * weight
        total_weight += weight
    
    return (weighted_sum / total_weight) * 100 if total_weight > 0 else 0.0

def aggregate_all_results():
    """Aggregate all results into a single JSON file"""
    
    if not RESULTS_DIR.exists():
        print("❌ Results directory not found")
        return
    
    all_results = []
    
    for result_dir in RESULTS_DIR.iterdir():
        if result_dir.is_dir():
            model_data = load_result(result_dir)
            if model_data:
                all_results.append(model_data)
    
    # Sort by average score
    all_results.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Add ranks
    for i, model in enumerate(all_results, 1):
        model['rank'] = i
    
    # Save to JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Aggregated {len(all_results)} model results")
    print(f"📊 Saved to {OUTPUT_FILE}")
    
    # Print summary
    print("\n🏆 Top 5 Models:")
    for model in all_results[:5]:
        print(f"  {model['rank']}. {model['name']}: {model['avg_score']:.1f}")

if __name__ == "__main__":
    aggregate_all_results()