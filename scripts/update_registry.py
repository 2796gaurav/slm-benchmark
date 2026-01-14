#!/usr/bin/env python3
"""Update model registry with new marketplace benchmark results."""

import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime


def _average_score(score_dict):
    """Calculate mean score from dictionary values."""
    if not score_dict:
        return 0.0
    
    values = []
    for v in score_dict.values():
        if isinstance(v, dict) and 'acc' in v:
            values.append(v['acc'] * 100)
        elif isinstance(v, dict) and 'score' in v:
            values.append(v['score'])
        elif isinstance(v, (int, float)):
            values.append(float(v))
    
    return sum(values) / len(values) if values else 0.0


def update_registry(submission_file: str, results_file: str, registry_file: str):
    """Update registry with new model using marketplace schema."""
    
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
    
    model = submission['model']
    
    # Build use_cases from results
    use_cases = {}
    for uc in ['rag', 'function_calling', 'coding', 'reasoning', 'guardrails']:
        uc_results = results.get(f'{uc}_scores', results.get(uc, {}))
        if uc_results:
            score = _average_score(uc_results)
            use_cases[uc] = {
                'score': score,
                'benchmarks': uc_results if isinstance(uc_results, dict) else {},
                'recommended': score >= 70.0  # Recommend if score >= 70
            }
    
    # Build performance metrics
    performance = results.get('performance', {})
    if not performance:
        performance = {
            'hardware': 'GitHub Actions (2-core CPU)',
            'quantizations': {}
        }
    
    # Build safety metrics
    safety = {
        'guardrails_compatible': True,
        'hallucination_rate': results.get('hallucination_rate', 0),
        'bias_score': results.get('bias_score', 'N/A'),
        'toxicity_rate': results.get('toxicity_rate', 0)
    }
    
    # Calculate aggregate score (weighted average of use cases)
    weights = {
        'rag': 0.25,
        'function_calling': 0.20,
        'coding': 0.15,
        'reasoning': 0.25,
        'guardrails': 0.15
    }
    
    aggregate_score = sum(
        use_cases.get(uc, {}).get('score', 0) * weights.get(uc, 0)
        for uc in weights.keys()
    )
    
    # Determine best_for and not_for
    best_for = [uc.replace('_', ' ').title() for uc, data in use_cases.items() if data.get('recommended')]
    not_for = [uc.replace('_', ' ').title() for uc, data in use_cases.items() 
               if data.get('score', 0) < 50]
    
    # Create model entry with marketplace schema
    model_entry = {
        'id': model.get('name', model['hf_repo']).lower().replace(' ', '-').replace('/', '-'),
        'name': model.get('name', model['hf_repo'].split('/')[-1]),
        'family': model.get('family', model['hf_repo'].split('/')[0]),
        'hf_repo': model['hf_repo'],
        'parameters': model.get('parameters', 'N/A'),
        'license': model.get('license', 'Unknown'),
        'context_window': model.get('context_window', results.get('context_window', 4096)),
        'architecture': model.get('architecture', results.get('architecture', 'Unknown')),
        
        'use_cases': use_cases,
        
        'domain_scores': {
            'general': aggregate_score,
            'finance': None,
            'healthcare': None,
            'legal': None
        },
        
        'performance': performance,
        'safety': safety,
        
        'compliance': {
            'hipaa': False,
            'gdpr': True,
            'finra': False
        },
        
        'tags': model.get('tags', []),
        'deployment_targets': model.get('deployment_targets', []),
        'best_for': best_for,
        'not_for': not_for,
        
        'aggregate_score': aggregate_score,
        'date_added': datetime.now().isoformat()[:10],
        'submitted_by': model.get('submitted_by', 'unknown')
    }
    
    # Add or update in registry
    model_id = model_entry['id']
    existing_idx = next((i for i, m in enumerate(registry['models']) if m['id'] == model_id), -1)
    
    if existing_idx >= 0:
        registry['models'][existing_idx] = model_entry
        print(f"🔄 Updated {model_entry['name']} in registry")
    else:
        registry['models'].append(model_entry)
        print(f"✅ Added {model_entry['name']} to registry")
    
    # Re-rank models by aggregate score
    registry['models'].sort(key=lambda x: x.get('aggregate_score', 0), reverse=True)
    for idx, m in enumerate(registry['models']):
        m['rank'] = idx + 1
    
    # Update metadata
    registry['metadata'] = {
        'last_updated': datetime.now().isoformat(),
        'total_models': len(registry['models']),
        'schema_version': '2.0.0',
        'platform': 'SLM Marketplace'
    }
    
    # Save
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Registry updated. {model_entry['name']} is now Rank #{model_entry.get('rank', '?')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update SLM Marketplace registry')
    parser.add_argument('--submission', required=True, help='Path to submission YAML')
    parser.add_argument('--results', required=True, help='Path to benchmark results JSON')
    parser.add_argument('--registry', required=True, help='Path to registry JSON')
    args = parser.parse_args()
    
    update_registry(args.submission, args.results, args.registry)