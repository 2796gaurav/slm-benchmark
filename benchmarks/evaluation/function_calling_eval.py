#!/usr/bin/env python3
"""
Function calling benchmarks for SLM Marketplace.
Evaluates models on tool use and API calling tasks using BFCL (Berkeley Function Calling Leaderboard).
"""

import logging
import json
import re
import torch
import subprocess
import sys
from typing import Dict, List

logger = logging.getLogger(__name__)


class FunctionCallingEvaluator:
    """
    Evaluates models on function calling benchmarks:
    - BFCL (Berkeley Function Calling Leaderboard)
    - NESTful
    - DispatchQA
    - Custom scenarios (single, multiple, parallel, multi-turn)
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self._check_model_interface()
    
    def _check_model_interface(self):
        """Determine model interface type"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'generate'):
            self.model_obj = self.model.model
            self.is_hflm = True
        elif hasattr(self.model, 'generate'):
            self.model_obj = self.model
            self.is_hflm = False
        else:
            raise ValueError("Model must have generate() method")
    
    def evaluate_all(self) -> Dict:
        """Run all function calling benchmarks"""
        logger.info("Running function calling benchmarks...")
        
        results = {}
        
        # BFCL - Berkeley Function Calling Leaderboard
        logger.info("  Running BFCL test...")
        results['bfcl_v2'] = self._run_bfcl_benchmark()
        
        # Test scenarios
        scenarios = [
            'single_function',
            'multiple_functions',
            'parallel_functions',
            'multi_turn'
        ]
        
        for scenario in scenarios:
            logger.info(f"  Running {scenario} test...")
            results[f'{scenario}_accuracy'] = self._test_function_scenario(scenario)
        
        # Compute aggregate score
        results['aggregate_score'] = self._compute_function_calling_score(results)
        
        return results
    
    def _check_bfcl_installed(self) -> bool:
        """Check if bfcl-eval is installed"""
        try:
            import bfcl_eval
            return True
        except ImportError:
            return False
    
    def _install_bfcl(self):
        """Install bfcl-eval package"""
        try:
            logger.info("Installing bfcl-eval package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bfcl-eval==2025.12.17"])
            return True
        except Exception as e:
            logger.warning(f"Failed to install bfcl-eval: {e}")
            return False
    
    def _run_bfcl_benchmark(self) -> Dict:
        """
        Berkeley Function Calling Leaderboard benchmark.
        
        Uses BFCL V3, BFCL Live, and BFCL Multi-Turn evaluations.
        Tests: Single, Multiple, Parallel, Parallel Multiple, Relevance, Multi-Turn
        """
        # Try to use BFCL library if available
        if not self._check_bfcl_installed():
            logger.warning("bfcl-eval not installed. Installing...")
            if not self._install_bfcl():
                logger.warning("Falling back to simplified BFCL test")
                return self._run_bfcl_simplified()
        
        try:
            from bfcl_eval import evaluate
            
            results = {}
            
            # BFCL V3 - All categories
            logger.info("  Running BFCL V3 (all categories)...")
            try:
                bfcl_v3_results = evaluate(
                    model=self.config.hf_repo,
                    test_category='all',
                    output_dir=None
                )
                results['bfcl_v3'] = {
                    'simple': bfcl_v3_results.get('simple', {}).get('accuracy', 0.0),
                    'multiple': bfcl_v3_results.get('multiple', {}).get('accuracy', 0.0),
                    'parallel': bfcl_v3_results.get('parallel', {}).get('accuracy', 0.0),
                    'parallel_multiple': bfcl_v3_results.get('parallel_multiple', {}).get('accuracy', 0.0),
                    'relevance': bfcl_v3_results.get('relevance', {}).get('accuracy', 0.0),
                    'overall': bfcl_v3_results.get('overall_accuracy', 0.0)
                }
            except Exception as e:
                logger.warning(f"BFCL V3 evaluation failed: {e}")
                results['bfcl_v3'] = {'overall': 0.0, 'error': str(e)}
            
            # BFCL Live - Real-world API calls
            logger.info("  Running BFCL Live...")
            try:
                bfcl_live_results = evaluate(
                    model=self.config.hf_repo,
                    test_category='live',
                    output_dir=None
                )
                results['bfcl_live'] = {
                    'executable_accuracy': bfcl_live_results.get('executable_accuracy', 0.0),
                    'ast_accuracy': bfcl_live_results.get('ast_accuracy', 0.0)
                }
            except Exception as e:
                logger.warning(f"BFCL Live evaluation failed: {e}")
                results['bfcl_live'] = {'executable_accuracy': 0.0, 'error': str(e)}
            
            # BFCL Multi-Turn
            logger.info("  Running BFCL Multi-Turn...")
            try:
                bfcl_multiturn_results = evaluate(
                    model=self.config.hf_repo,
                    test_category='multi_turn',
                    output_dir=None
                )
                results['bfcl_multiturn'] = {
                    'accuracy': bfcl_multiturn_results.get('accuracy', 0.0),
                    'conversation_success_rate': bfcl_multiturn_results.get('conversation_success_rate', 0.0)
                }
            except Exception as e:
                logger.warning(f"BFCL Multi-Turn evaluation failed: {e}")
                results['bfcl_multiturn'] = {'accuracy': 0.0, 'error': str(e)}
            
            # Compute aggregate
            scores = []
            if 'bfcl_v3' in results and 'overall' in results['bfcl_v3']:
                scores.append(results['bfcl_v3']['overall'] * 100)
            if 'bfcl_live' in results and 'executable_accuracy' in results['bfcl_live']:
                scores.append(results['bfcl_live']['executable_accuracy'] * 100)
            if 'bfcl_multiturn' in results and 'accuracy' in results['bfcl_multiturn']:
                scores.append(results['bfcl_multiturn']['accuracy'] * 100)
            
            results['aggregate_accuracy'] = sum(scores) / len(scores) if scores else 0.0
            
            return results
            
        except Exception as e:
            logger.warning(f"BFCL library evaluation failed: {e}")
            return self._run_bfcl_simplified()
    
    def _run_bfcl_simplified(self) -> Dict:
        """
        Simplified BFCL test cases (fallback when library not available).
        
        Tests model's ability to call functions with correct parameters.
        """
        # Simplified BFCL test cases
        test_cases = [
            {
                'functions': [
                    {
                        'name': 'get_weather',
                        'description': 'Get weather for a location',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'location': {'type': 'string'},
                                'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}
                            },
                            'required': ['location']
                        }
                    }
                ],
                'prompt': 'What is the weather in Paris?',
                'expected': {
                    'function': 'get_weather',
                    'arguments': {'location': 'Paris'}
                }
            },
            {
                'functions': [
                    {
                        'name': 'calculate',
                        'description': 'Perform a calculation',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'expression': {'type': 'string'}
                            },
                            'required': ['expression']
                        }
                    }
                ],
                'prompt': 'Calculate 15 * 23',
                'expected': {
                    'function': 'calculate',
                    'arguments': {'expression': '15 * 23'}
                }
            },
            {
                'functions': [
                    {
                        'name': 'search',
                        'description': 'Search the web',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'query': {'type': 'string'},
                                'limit': {'type': 'integer'}
                            },
                            'required': ['query']
                        }
                    }
                ],
                'prompt': 'Search for information about quantum computing',
                'expected': {
                    'function': 'search',
                    'arguments': {'query': 'quantum computing'}
                }
            }
        ]
        
        correct = 0
        total = 0
        
        for test_case in test_cases:
            try:
                # Format function calling prompt
                functions_json = json.dumps(test_case['functions'], indent=2)
                prompt = f"""You have access to the following functions:

{functions_json}

User: {test_case['prompt']}

Call the appropriate function with the correct arguments. Format your response as a function call."""
                
                # Generate response
                response = self._generate_answer(prompt)
                
                # Check if response contains function call
                expected_func = test_case['expected']['function']
                response_lower = response.lower()
                
                # Check for function name
                if expected_func.lower() in response_lower:
                    # Check for expected arguments
                    expected_args = test_case['expected']['arguments']
                    args_found = 0
                    for arg_name, arg_value in expected_args.items():
                        if arg_name.lower() in response_lower or str(arg_value).lower() in response_lower:
                            args_found += 1
                    
                    # Consider correct if function name and at least one argument found
                    if args_found > 0:
                        correct += 1
                
                total += 1
                
            except Exception as e:
                logger.warning(f"BFCL test case failed: {e}")
                total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'note': 'Simplified BFCL test (install bfcl-eval==2025.12.17 for full evaluation)'
        }
    
    def _test_function_scenario(self, scenario: str) -> float:
        """Test specific function calling scenario"""
        scenarios = {
            'single_function': {
                'functions': [{'name': 'get_time', 'description': 'Get current time'}],
                'prompt': 'What time is it?',
                'expected_function': 'get_time'
            },
            'multiple_functions': {
                'functions': [
                    {'name': 'search', 'description': 'Search the web'},
                    {'name': 'get_weather', 'description': 'Get weather for a location'}
                ],
                'prompt': 'Search for weather in New York',
                'expected_function': 'search'
            },
            'parallel_functions': {
                'functions': [
                    {'name': 'get_stock_price', 'description': 'Get stock price for a symbol'},
                    {'name': 'get_news', 'description': 'Get news articles'}
                ],
                'prompt': 'Get the stock price of AAPL and recent news',
                'expected_functions': ['get_stock_price', 'get_news']
            },
            'multi_turn': {
                'functions': [{'name': 'add_to_cart', 'description': 'Add item to shopping cart'}],
                'prompt': 'Add a laptop to my cart',
                'expected_function': 'add_to_cart'
            }
        }
        
        if scenario not in scenarios:
            return 0.0
        
        test = scenarios[scenario]
        functions_json = json.dumps(test['functions'], indent=2)
        prompt = f"""You have access to the following functions:

{functions_json}

User: {test['prompt']}

Call the appropriate function."""
        
        try:
            response = self._generate_answer(prompt)
            response_lower = response.lower()
            
            # Check accuracy
            if scenario == 'parallel_functions':
                expected = test['expected_functions']
                correct = sum(1 for func in expected if func.lower() in response_lower)
                return (correct / len(expected) * 100) if expected else 0.0
            else:
                expected = test['expected_function']
                return 100.0 if expected.lower() in response_lower else 0.0
        except Exception as e:
            logger.warning(f"{scenario} test failed: {e}")
            return 0.0
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using model"""
        try:
            if self.is_hflm:
                output = self.model_obj.generate(prompt, max_tokens=150)
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.config.hf_repo, trust_remote_code=True)
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model_obj.generate(**inputs, max_new_tokens=150)
                output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if isinstance(output, str):
                return output
            elif isinstance(output, (list, tuple)):
                return ' '.join(str(x) for x in output)
            else:
                return str(output)
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            return ""
    
    def _compute_function_calling_score(self, results: Dict) -> float:
        """Compute aggregate function calling score"""
        scores = []
        
        # BFCL accuracy
        if 'bfcl_v2' in results:
            scores.append(results['bfcl_v2'].get('accuracy', 0))
        
        # Scenario accuracies
        for key, value in results.items():
            if key.endswith('_accuracy') and isinstance(value, (int, float)):
                scores.append(value)
        
        return sum(scores) / len(scores) if scores else 0.0
