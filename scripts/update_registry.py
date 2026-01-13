#!/usr/bin/env python3
"""Update model registry with new results"""

import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

def _average_score(score_dict):
    """Calculate mean score from dictionary values"""
    if not score_dict:
        return 0.0
    
    values = []
    for v in score_dict.values():
        if isinstance(v, dict) and 'acc' in v:
            values.append(v['acc'] * 100)
        elif isinstance(v, (int, float)):
            values.append(float(v))
    
    return sum(values) / len(values) if values else 0.0

def update_registry(submission_file: str, results_file: str, registry_file: str):
    # Load submission
    with open(submission_file, 'r') as f:
        submission = yaml.safe_load(f)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load or create registry
    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    except FileNotFoundError:
        registry = {'models': [], 'metadata': {}}
    
    # Create model entry
    model = submission['model']
    model_entry = {
        'id': model['name'].lower().replace(' ', '-'),
        'name': model['name'],
        'family': model['family'],
        'hf_repo': model['hf_repo'],
        'parameters': model['parameters'],
        'license': model['license'],
        'aggregate_score': results['aggregate_score'],
        'scores': {
            'reasoning': _average_score(results.get('reasoning_scores', {})),
            'coding': _average_score(results.get('coding_scores', {})),
            'math': _average_score(results.get('math_scores', {})),
            'language': _average_score(results.get('language_scores', {})),
            'edge': _average_score(results.get('edge_metrics', {})),
            'safety': _average_score(results.get('safety_scores', {}))
        },
        'quantizations': model['quantizations'],
        'categories': model.get('categories', []),
        'date_added': datetime.now().isoformat()[:10],
        'submitted_by': model.get('submitted_by', 'unknown')
    }
    
    # Add or update in registry
    model_id = model_entry['id']
    existing_idx = next((i for i, m in enumerate(registry['models']) if m['id'] == model_id), -1)
    
    if existing_idx >= 0:
        registry['models'][existing_idx] = model_entry
        print(f"🔄 Updated {model['name']} in registry")
    else:
        registry['models'].append(model_entry)
        print(f"✅ Added {model['name']} to registry")
    
    # Re-rank models
    registry['models'].sort(key=lambda x: x['aggregate_score'], reverse=True)
    for idx, m in enumerate(registry['models']):
        m['rank'] = idx + 1
    
    # Update metadata
    registry['metadata'] = {
        'last_updated': datetime.now().isoformat(),
        'total_models': len(registry['models'])
    }
    
    # Save
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Added {model['name']} to registry (Rank #{model_entry['rank']})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--registry', required=True)
    args = parser.parse_args()
    
    update_registry(args.submission, args.results, args.registry)