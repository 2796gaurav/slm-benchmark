import json
import os
def _average_score(score_dict):
    """Calculate mean score from dictionary values, handling lm-eval style metrics."""
    if not score_dict:
        return 0.0
    
    values = []
    for v in score_dict.values():
        if isinstance(v, dict):
            acc_norm = v.get('acc_norm,none', v.get('acc_norm'))
            acc = v.get('acc,none', v.get('acc'))
            
            if acc_norm is not None:
                values.append(float(acc_norm) * 100)
            elif acc is not None:
                values.append(float(acc) * 100)
            else:
                for k, val in v.items():
                    if 'acc' in k and isinstance(val, (int, float)):
                        values.append(float(val) * 100)
                        break
        elif isinstance(v, (int, float)):
            values.append(float(v))
    
    return sum(values) / len(values) if values else 0.0

def recalculate_registry(registry_path):
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    weights = {
        'rag': 0.25,
        'function_calling': 0.20,
        'coding': 0.15,
        'reasoning': 0.25,
        'guardrails': 0.15
    }
    
    for model in registry['models']:
        use_cases = model.get('use_cases', {})
        for uc, data in use_cases.items():
            if isinstance(data, dict) and 'benchmarks' in data:
                new_score = _average_score(data['benchmarks'])
                data['score'] = new_score
                data['recommended'] = new_score >= 70.0
        
        # Calculate aggregate score
        aggregate_score = sum(
            use_cases.get(uc, {}).get('score', 0) * weights.get(uc, 0)
            for uc in weights.keys()
        )
        model['aggregate_score'] = round(aggregate_score, 2)
        
        # Update best_for/not_for
        model['best_for'] = [uc.replace('_', ' ').title() for uc, data in use_cases.items() if data.get('recommended')]
        model['not_for'] = [uc.replace('_', ' ').title() for uc, data in use_cases.items() if data.get('score', 0) < 50]

    # Re-rank
    registry['models'].sort(key=lambda x: x.get('aggregate_score', 0), reverse=True)
    for idx, m in enumerate(registry['models']):
        m['rank'] = idx + 1
        
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"✅ Recalculated scores for {len(registry['models'])} models in {registry_path}")

if __name__ == "__main__":
    recalculate_registry('models/registry.json')
