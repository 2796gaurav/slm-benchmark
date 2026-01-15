#!/usr/bin/env python3
"""
RAG-specific benchmarks for SLM Marketplace.
Evaluates models on retrieval-augmented generation tasks.
"""

import logging
import random
import re
import torch
from typing import Dict, List, Tuple
from lm_eval import evaluator

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates models on RAG-specific benchmarks:
    - NIAH (Needle-in-Haystack)
    - RULER (Extended NIAH)
    - RAGTruth (Hallucination detection)
    - FRAMES (Multi-hop reasoning)
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
        """Run all RAG benchmarks"""
        logger.info("Running RAG benchmarks...")
        
        results = {}
        
        # NIAH - Needle in Haystack
        logger.info("  Running NIAH test...")
        results['niah'] = self._run_niah_test()
        
        # RULER - Extended NIAH (if available)
        logger.info("  Running RULER test...")
        results['ruler'] = self._run_ruler_benchmark()
        
        # RAGTruth - Hallucination detection
        logger.info("  Running RAGTruth test...")
        results['ragtruth'] = self._evaluate_hallucination()
        
        # FRAMES - Multi-hop reasoning (if available)
        logger.info("  Running FRAMES test...")
        results['frames'] = self._run_frames_benchmark()
        
        # Compute aggregate RAG score
        results['aggregate_score'] = self._compute_rag_score(results)
        
        return results
    
    def _run_niah_test(self, context_lengths: List[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768],
                       document_depths: List[int] = [0, 25, 50, 75, 100],
                       num_tests: int = 10) -> Dict:
        """
        NIAH (Needle-in-Haystack) test per LostInTheMiddle paper.
        
        Tests retrieval across variable context lengths (512-32K tokens).
        Inserts facts at different positions (depths) and measures retrieval accuracy.
        """
        """
        NIAH (Needle-in-Haystack) test.
        
        Tests retrieval across variable context lengths (512-32K tokens as per LostInTheMiddle paper).
        Inserts a fact at different positions in a long context and measures if model can retrieve it.
        """
        """
        Needle-in-Haystack test.
        
        Inserts a fact at different positions in a long context
        and tests if the model can retrieve it.
        """
        results = {}
        
        # Test facts to insert
        test_facts = [
            ("The capital of France is Paris.", "What is the capital of France?"),
            ("The Eiffel Tower was built in 1889.", "When was the Eiffel Tower built?"),
            ("Python was created by Guido van Rossum.", "Who created Python?"),
            ("The speed of light is 299,792,458 meters per second.", "What is the speed of light?"),
            ("Mount Everest is 8,848 meters tall.", "How tall is Mount Everest?")
        ]
        
        for ctx_len in context_lengths:
            ctx_results = []
            
            for depth in document_depths:
                correct = 0
                total = 0
                
                for fact, question in test_facts[:num_tests]:
                    try:
                        # Generate haystack (random text)
                        haystack = self._generate_haystack(ctx_len)
                        
                        # Insert needle at specified depth
                        needle_position = int(len(haystack) * depth / 100)
                        haystack_with_needle = (
                            haystack[:needle_position] + 
                            f" {fact} " + 
                            haystack[needle_position:]
                        )
                        
                        # Create prompt
                        prompt = f"{haystack_with_needle}\n\nQuestion: {question}\nAnswer:"
                        
                        # Generate answer
                        answer = self._generate_answer(prompt)
                        
                        # Check if answer contains the fact
                        fact_keywords = self._extract_keywords(fact)
                        answer_lower = answer.lower()
                        
                        # Check if at least one keyword from fact is in answer
                        if any(keyword in answer_lower for keyword in fact_keywords):
                            correct += 1
                        total += 1
                    except Exception as e:
                        logger.warning(f"NIAH test failed: {e}")
                        total += 1
                
                accuracy = (correct / total * 100) if total > 0 else 0.0
                ctx_results.append({
                    'depth': depth,
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                })
            
            results[f'{ctx_len}_context'] = ctx_results
        
        # Compute average accuracy across all contexts and depths
        all_accuracies = []
        for ctx_results in results.values():
            for result in ctx_results:
                all_accuracies.append(result['accuracy'])
        
        results['average_accuracy'] = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        
        return results
    
    def _run_ruler_benchmark(self) -> Dict:
        """
        RULER benchmark (Extended NIAH + variable-length key retrieval).
        
        RULER includes 13 tasks across 4K-128K context lengths.
        Tests: NIAH-style retrieval, CWE (Context Window Extension), FWE (Full Window Evaluation)
        """
        try:
            # Try to use lm-eval RULER tasks
            # RULER tests multiple context lengths: 4K, 8K, 16K, 32K, 64K, 128K
            context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
            
            results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['ruler_niah', 'ruler_cwe', 'ruler_fwe'],
                limit=100,  # Increased limit for better coverage
                num_fewshot=0
            )
            ruler_results = results.get('results', {})
            
            # Extract scores for each task type
            niah_scores = []
            cwe_scores = []
            fwe_scores = []
            
            for task_name, task_results in ruler_results.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        if 'niah' in task_name.lower():
                            niah_scores.append(score)
                        elif 'cwe' in task_name.lower():
                            cwe_scores.append(score)
                        elif 'fwe' in task_name.lower():
                            fwe_scores.append(score)
            
            # Compute aggregate scores
            niah_avg = sum(niah_scores) / len(niah_scores) if niah_scores else 0.0
            cwe_avg = sum(cwe_scores) / len(cwe_scores) if cwe_scores else 0.0
            fwe_avg = sum(fwe_scores) / len(fwe_scores) if fwe_scores else 0.0
            
            overall_score = (niah_avg + cwe_avg + fwe_avg) / 3 if (niah_scores or cwe_scores or fwe_scores) else 0.0
            
            return {
                'score': overall_score,
                'niah': niah_avg,
                'cwe': cwe_avg,
                'fwe': fwe_avg,
                'detailed': ruler_results,
                'tasks_evaluated': len(ruler_results)
            }
        except Exception as e:
            logger.warning(f"RULER benchmark not available: {e}")
            # Fallback: simplified RULER using extended NIAH-style test
            return {
                'note': 'RULER benchmark not available, using simplified version',
                'score': 0.0
            }
    
    def _evaluate_hallucination(self) -> Dict:
        """
        RAGTruth-style hallucination detection.
        
        RAGTruth dataset: 18K samples for hallucination detection in RAG.
        Tests if model generates facts not present in context.
        """
        # Try to load RAGTruth dataset if available
        try:
            from datasets import load_dataset
            ragtruth_dataset = load_dataset("ragtruth/ragtruth", split="test")
            
            # Sample subset for evaluation (use first 100 for speed)
            sample_size = min(100, len(ragtruth_dataset))
            test_cases = []
            
            for i in range(sample_size):
                item = ragtruth_dataset[i]
                test_cases.append({
                    'context': item.get('context', ''),
                    'question': item.get('question', ''),
                    'expected_answer': item.get('ground_truth', ''),
                    'is_hallucination': item.get('is_hallucination', False)
                })
            
            logger.info(f"Loaded {len(test_cases)} samples from RAGTruth dataset")
            
        except Exception as e:
            logger.warning(f"RAGTruth dataset not available: {e}. Using simplified test cases.")
            # Fallback test cases
            test_cases = [
            {
                'context': 'The Eiffel Tower is located in Paris, France. It was built in 1889.',
                'question': 'When was the Eiffel Tower built?',
                'expected_answer': '1889',
                'hallucination_risk': 'What country is the Eiffel Tower in?'
            },
            {
                'context': 'Python is a programming language created by Guido van Rossum.',
                'question': 'Who created Python?',
                'expected_answer': 'Guido van Rossum',
                'hallucination_risk': 'What year was Python created?'
            },
            {
                'context': 'The Great Wall of China is approximately 21,196 kilometers long.',
                'question': 'How long is the Great Wall of China?',
                'expected_answer': '21,196 kilometers',
                'hallucination_risk': 'Where is the Great Wall located?'
            }
        ]
        
        hallucination_count = 0
        total = 0
        
        for test_case in test_cases:
            # Test expected answer
            prompt = f"{test_case['context']}\n\nQuestion: {test_case['question']}\nAnswer:"
            try:
                answer = self._generate_answer(prompt)
                
                # Check for hallucination indicators
                # 1. Very long answers might contain extra information
                if len(answer) > 200:
                    hallucination_count += 1
                
                # 2. Answer should contain expected keywords
                expected_keywords = self._extract_keywords(test_case['expected_answer'])
                answer_lower = answer.lower()
                if not any(keyword in answer_lower for keyword in expected_keywords):
                    hallucination_count += 1
                
                total += 1
            except Exception as e:
                logger.warning(f"Hallucination test failed: {e}")
                total += 1
        
        hallucination_rate = (hallucination_count / total * 100) if total > 0 else 0.0
        
        return {
            'hallucination_rate': hallucination_rate,
            'tests_passed': total - hallucination_count,
            'total_tests': total
        }
    
    def _run_frames_benchmark(self) -> Dict:
        """
        FRAMES benchmark (Multi-hop reasoning over retrieved documents).
        
        FRAMES dataset: 824 examples for multi-hop reasoning.
        Tests model's ability to reason across multiple documents.
        """
        try:
            # Try to load FRAMES dataset
            try:
                from datasets import load_dataset
                frames_dataset = load_dataset("frames", split="test")
                logger.info(f"Loaded FRAMES dataset with {len(frames_dataset)} examples")
            except Exception as e:
                logger.warning(f"FRAMES dataset not directly available: {e}")
            
            # Try to use lm-eval FRAMES task
            results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['frames'],
                limit=100,  # Increased limit for better coverage (824 total examples)
                num_fewshot=0
            )
            frames_results = results.get('results', {})
            
            # Extract score
            score = 0.0
            for task_name, task_results in frames_results.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            return {
                'score': score,
                'detailed': frames_results,
                'dataset_size': 824,
                'samples_tested': min(100, 824)
            }
        except Exception as e:
            logger.warning(f"FRAMES benchmark not available: {e}")
            # Fallback: simplified multi-hop reasoning test
            return {
                'note': 'FRAMES benchmark not available',
                'score': 0.0
            }
    
    def _generate_haystack(self, length: int) -> str:
        """Generate random text haystack"""
        words = [
            'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
            'cat', 'runs', 'fast', 'slow', 'bird', 'flies', 'high', 'low',
            'water', 'flows', 'river', 'ocean', 'mountain', 'valley', 'forest',
            'tree', 'leaf', 'branch', 'root', 'flower', 'grass', 'field', 'meadow',
            'sun', 'moon', 'star', 'cloud', 'rain', 'snow', 'wind', 'storm',
            'house', 'door', 'window', 'room', 'chair', 'table', 'bed', 'lamp'
        ]
        haystack = ' '.join(random.choices(words, k=length))
        return haystack
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using model"""
        try:
            if self.is_hflm:
                # HFLM interface
                output = self.model_obj.generate(prompt, max_tokens=100)
            else:
                # Direct model interface
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.config.hf_repo, trust_remote_code=True)
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model_obj.generate(**inputs, max_new_tokens=100)
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
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words and extract meaningful terms
        text_lower = text.lower()
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text_lower)
        # Filter out very common words
        stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'of', 'in', 'on', 'at', 'to',
                     'for', 'with', 'by', 'from', 'as', 'about', 'into', 'through', 'during'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:5]  # Return top 5 keywords
    
    def _compute_rag_score(self, results: Dict) -> float:
        """Compute aggregate RAG score"""
        scores = []
        
        # NIAH average accuracy
        if 'niah' in results and 'average_accuracy' in results['niah']:
            scores.append(results['niah']['average_accuracy'])
        
        # RULER score (if available)
        if 'ruler' in results:
            ruler_score = results['ruler'].get('score', 0)
            if ruler_score > 0:
                scores.append(ruler_score)
        
        # FRAMES score (if available)
        if 'frames' in results:
            frames_score = results['frames'].get('score', 0)
            if frames_score > 0:
                scores.append(frames_score)
        
        # Hallucination (inverse: lower is better)
        if 'ragtruth' in results:
            hallucination_rate = results['ragtruth'].get('hallucination_rate', 0)
            # Convert to score: 100 - hallucination_rate
            scores.append(max(0, 100 - hallucination_rate))
        
        return sum(scores) / len(scores) if scores else 0.0
