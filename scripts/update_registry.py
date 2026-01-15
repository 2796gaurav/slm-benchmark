#!/usr/bin/env python3
"""Update model registry with new marketplace benchmark results."""

import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime


def _determine_deployment_targets(performance):
    """Determine deployment targets based on RAM requirements"""
    deployment_targets = []
    
    # Get minimum RAM across all quantizations
    quantizations = performance.get('quantizations', {})
    if quantizations:
        min_ram = min([q.get('ram_gb', 999) for q in quantizations.values()], default=999)
        
        if min_ram <= 2:
            deployment_targets.append('raspberry_pi_5')
        if min_ram <= 4:
            deployment_targets.append('jetson_orin')
        if min_ram <= 8:
            deployment_targets.append('16gb_laptop')
        if min_ram <= 16:
            deployment_targets.append('mobile_high_end')
        if min_ram > 16:
            deployment_targets.append('edge_server')
    else:
        # Default if no performance data
        deployment_targets = ['edge_server']
    
    return deployment_targets


def _average_score(score_dict):
    """Calculate mean score from dictionary values, handling lm-eval style metrics."""
    if not score_dict:
        return 0.0
    
    values = []
    for v in score_dict.values():
        if isinstance(v, dict):
            # Handle lm-eval style metrics (acc,none, acc_norm,none, etc.)
            # Priority: acc_norm > acc > any key starting with acc
            acc_norm = v.get('acc_norm,none', v.get('acc_norm'))
            acc = v.get('acc,none', v.get('acc'))
            
            if acc_norm is not None:
                values.append(float(acc_norm) * 100)
            elif acc is not None:
                values.append(float(acc) * 100)
            else:
                # Fallback: check for any key containing 'acc'
                for k, val in v.items():
                    if 'acc' in k and isinstance(val, (int, float)):
                        values.append(float(val) * 100)
                        break
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
    
    # Build use_cases from results (enhanced)
    use_cases = {}
    
    # RAG
    rag_results = results.get('rag_scores', {})
    if rag_results:
        rag_score = rag_results.get('aggregate_score', 0.0)
        use_cases['rag'] = {
            'score': rag_score,
            'benchmarks': {
                'niah': rag_results.get('niah', {}).get('average_accuracy', 0.0),
                'ruler': rag_results.get('ruler', {}).get('score', 0.0),
                'ragtruth': 100 - rag_results.get('ragtruth', {}).get('hallucination_rate', 0.0),
                'frames': rag_results.get('frames', {}).get('score', 0.0)
            },
            'recommended': rag_score >= 70.0
        }
    
    # Function Calling
    fc_results = results.get('function_calling_scores', {})
    if fc_results:
        fc_score = fc_results.get('aggregate_score', 0.0)
        use_cases['function_calling'] = {
            'score': fc_score,
            'benchmarks': {
                'bfcl_v2': fc_results.get('bfcl_v2', {}).get('accuracy', 0.0),
                'single_function': fc_results.get('single_function_accuracy', 0.0),
                'multiple_functions': fc_results.get('multiple_functions_accuracy', 0.0),
                'parallel_functions': fc_results.get('parallel_functions_accuracy', 0.0),
                'multi_turn': fc_results.get('multi_turn_accuracy', 0.0)
            },
            'recommended': fc_score >= 70.0
        }
    
    # Coding
    coding_results = results.get('coding_scores', {})
    if coding_results:
        coding_score = _average_score(coding_results)
        use_cases['coding'] = {
            'score': coding_score,
            'benchmarks': coding_results,
            'recommended': coding_score >= 60.0
        }
    
    # Reasoning
    reasoning_results = results.get('reasoning_scores', {})
    if reasoning_results:
        reasoning_score = _average_score(reasoning_results)
        use_cases['reasoning'] = {
            'score': reasoning_score,
            'benchmarks': reasoning_results,
            'recommended': reasoning_score >= 60.0
        }
    
    # Guardrails
    guardrails_results = results.get('guardrails_scores', {})
    if guardrails_results:
        guardrails_score = guardrails_results.get('aggregate_score', 0.0)
        use_cases['guardrails'] = {
            'score': guardrails_score,
            'benchmarks': guardrails_results,
            'recommended': guardrails_score >= 70.0
        }
    
    # Build performance metrics (enhanced with CPU performance)
    performance = {
        'hardware': 'GitHub Actions (2-core CPU)',
        'quantizations': {}
    }
    
    # Extract CPU performance results
    cpu_perf = results.get('cpu_performance', {})
    if cpu_perf:
        # Get hardware spec
        hardware = cpu_perf.get('hardware', {})
        if hardware:
            performance['hardware'] = f"{hardware.get('platform', 'GitHub Actions')} ({hardware.get('cpu_cores', 2)}-core CPU)"
        
        # Extract TPS and TTFT for primary quantization
        tps_results = cpu_perf.get('tps', {})
        ttft_results = cpu_perf.get('ttft', {})
        memory_results = cpu_perf.get('memory', {})
        
        # Use 500 tokens and 2048 context as standard
        tps_500 = tps_results.get('500_tokens', {}).get('mean_tps', 0.0) if tps_results else 0.0
        ttft_2048 = ttft_results.get('2048_context', {}).get('mean_ttft_ms', 0.0) if ttft_results else 0.0
        ram_gb = memory_results.get('ram_gb', 0.0) if memory_results else 0.0
        
        # Get quantization from submission (default to FP16)
        quantizations = model.get('quantizations', [{'name': 'FP16'}])
        primary_quant = quantizations[0]['name'].lower() if quantizations else 'fp16'
        
        performance['quantizations'][primary_quant] = {
            'tps_output': tps_500,
            'ttft_ms': ttft_2048,
            'ram_gb': ram_gb
        }
    
    # Build safety metrics (enhanced)
    safety_scores = results.get('safety_scores', {})
    guardrails_scores = results.get('guardrails_scores', {})
    
    # Helper function for bias score
    def _compute_bias_score(safety_scores, guardrails_scores):
        """Compute bias score as letter grade"""
        bias_rate = 0.0
        if guardrails_scores and 'bias_assessment' in guardrails_scores:
            bias_rate = guardrails_scores['bias_assessment'].get('bias_rate', 0.0)
        elif safety_scores and 'bias' in safety_scores:
            bias_data = safety_scores['bias']
            if isinstance(bias_data, dict):
                bias_rate = bias_data.get('gender_bias_score', 0.0)
        
        # Convert to letter grade
        if bias_rate < 10:
            return 'A+'
        elif bias_rate < 20:
            return 'A'
        elif bias_rate < 30:
            return 'B+'
        elif bias_rate < 40:
            return 'B'
        elif bias_rate < 50:
            return 'C+'
        else:
            return 'C'
    
    safety = {
        'guardrails_compatible': True,
        'hallucination_rate': (
            guardrails_scores.get('hallucination_detection', {}).get('detection_rate', 0.0) if guardrails_scores
            else safety_scores.get('toxicity', {}).get('toxicity_rate', 0.0) if safety_scores
            else 0.0
        ),
        'bias_score': _compute_bias_score(safety_scores, guardrails_scores),
        'toxicity_rate': (
            safety_scores.get('toxicity', {}).get('toxicity_rate', 0.0) if safety_scores
            else guardrails_scores.get('toxicity_filtering', {}).get('filter_rate', 0.0) if guardrails_scores
            else 0.0
        )
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
            'finance': results.get('domain_scores', {}).get('finance', {}).get('aggregate_score') if results.get('domain_scores') else None,
            'healthcare': results.get('domain_scores', {}).get('healthcare', {}).get('aggregate_score') if results.get('domain_scores') else None,
            'legal': results.get('domain_scores', {}).get('legal', {}).get('aggregate_score') if results.get('domain_scores') else None,
            'science': results.get('domain_scores', {}).get('science', {}).get('aggregate_score') if results.get('domain_scores') else None
        },
        
        'performance': performance,
        'safety': safety,
        
        'compliance': {
            'hipaa': False,
            'gdpr': True,
            'finra': False
        },
        
        'tags': model.get('tags', []),
        'deployment_targets': _determine_deployment_targets(performance),
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