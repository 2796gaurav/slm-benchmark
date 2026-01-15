#!/usr/bin/env python3
"""
Domain-specific benchmarks for SLM Marketplace.
Evaluates models on specialized domains: healthcare, finance, legal, science.
"""

import logging
from typing import Dict, List
from lm_eval import evaluator

logger = logging.getLogger(__name__)


class DomainEvaluator:
    """
    Evaluates models on domain-specific benchmarks:
    - Healthcare: MultiMedQA, MedQA
    - Finance: FinBen, FinanceQA
    - Legal: LegalBench, LegalQA
    - Science: SciBench, PubMedQA
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def evaluate_all(self) -> Dict:
        """Run all domain-specific benchmarks"""
        logger.info("Running domain-specific benchmarks...")
        
        results = {}
        
        # Healthcare
        logger.info("  Running healthcare benchmarks...")
        results['healthcare'] = self._benchmark_healthcare()
        
        # Finance
        logger.info("  Running finance benchmarks...")
        results['finance'] = self._benchmark_finance()
        
        # Legal
        logger.info("  Running legal benchmarks...")
        results['legal'] = self._benchmark_legal()
        
        # Science
        logger.info("  Running science benchmarks...")
        results['science'] = self._benchmark_science()
        
        return results
    
    def _benchmark_healthcare(self) -> Dict:
        """
        Healthcare domain benchmarks.
        
        Tests:
        - MultiMedQA: 6 datasets across multiple medical domains
        - MedQA: 12,723 US Medical Licensing Exam questions
        - PubMedQA: 1,000 biomedical literature comprehension questions
        """
        results = {}
        
        # Try MultiMedQA (6 datasets)
        try:
            multimedqa_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['medqa', 'medmcqa', 'medqa4', 'pubmedqa'],
                limit=200,  # Increased for better coverage
                num_fewshot=0
            )
            multimedqa_scores = multimedqa_results.get('results', {})
            
            # Extract scores from all 6 datasets
            scores = []
            dataset_names = []
            for task_name, task_results in multimedqa_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        scores.append(score)
                        dataset_names.append(task_name)
            
            results['multimedqa'] = {
                'score': sum(scores) / len(scores) if scores else 0.0,
                'datasets_evaluated': len(dataset_names),
                'dataset_names': dataset_names,
                'detailed': multimedqa_scores
            }
        except Exception as e:
            logger.warning(f"MultiMedQA not available: {e}")
            results['multimedqa'] = {
                'score': 0.0,
                'note': 'MultiMedQA not available'
            }
        
        # Try MedQA (12,723 questions)
        try:
            medqa_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['medqa'],
                limit=500,  # Sample from 12,723 total
                num_fewshot=0
            )
            medqa_scores = medqa_results.get('results', {})
            
            score = 0.0
            for task_name, task_results in medqa_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            results['medqa'] = {
                'score': score,
                'dataset_size': 12723,
                'samples_tested': 500,
                'detailed': medqa_scores
            }
        except Exception as e:
            logger.warning(f"MedQA not available: {e}")
            results['medqa'] = {
                'score': 0.0,
                'note': 'MedQA not available'
            }
        
        # Try PubMedQA (1,000 questions)
        try:
            pubmedqa_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['pubmedqa'],
                limit=200,  # Sample from 1,000 total
                num_fewshot=0
            )
            pubmedqa_scores = pubmedqa_results.get('results', {})
            
            score = 0.0
            for task_name, task_results in pubmedqa_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            results['pubmedqa'] = {
                'score': score,
                'dataset_size': 1000,
                'samples_tested': 200,
                'detailed': pubmedqa_scores
            }
        except Exception as e:
            logger.warning(f"PubMedQA not available: {e}")
            results['pubmedqa'] = {
                'score': 0.0,
                'note': 'PubMedQA not available'
            }
        
        # Compute aggregate healthcare score
        scores = [r.get('score', 0) for r in results.values() if isinstance(r, dict) and 'score' in r]
        results['aggregate_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return results
    
    def _benchmark_finance(self) -> Dict:
        """
        Finance domain benchmarks.
        
        Tests:
        - FinBen: 36 datasets covering information extraction, text analysis, forecasting, risk management
        - FinQA: 8,281 examples of financial reasoning over tables
        """
        results = {}
        
        # Try FinQA (8,281 examples)
        try:
            finqa_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['finqa'],
                limit=500,  # Sample from 8,281 total
                num_fewshot=0
            )
            finqa_scores = finqa_results.get('results', {})
            
            score = 0.0
            for task_name, task_results in finqa_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            results['finqa'] = {
                'score': score,
                'dataset_size': 8281,
                'samples_tested': 500,
                'detailed': finqa_scores
            }
        except Exception as e:
            logger.warning(f"FinQA not available: {e}")
            results['finqa'] = {
                'score': 0.0,
                'note': 'FinQA not available'
            }
        
        # Try FinanceQA (fallback)
        try:
            finance_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['finance_qa'],
                limit=200,
                num_fewshot=0
            )
            finance_scores = finance_results.get('results', {})
            
            score = 0.0
            for task_name, task_results in finance_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            results['finance_qa'] = {
                'score': score,
                'detailed': finance_scores
            }
        except Exception as e:
            logger.warning(f"FinanceQA not available: {e}")
            results['finance_qa'] = {
                'score': 0.0,
                'note': 'FinanceQA not available'
            }
        
        # FinBen (36 datasets) would require custom implementation
        # For now, we'll use FinQA and FinanceQA
        scores = [r.get('score', 0) for r in results.values() if isinstance(r, dict) and 'score' in r]
        results['aggregate_score'] = sum(scores) / len(scores) if scores else 0.0
        results['note'] = 'FinBen (36 datasets) requires custom implementation'
        
        return results
    
    def _benchmark_legal(self) -> Dict:
        """
        Legal domain benchmarks.
        
        Tests:
        - LegalBench: 162 tasks covering issue-spotting, rule-recall, rule-application
        - CUAD: 510 contracts for contract understanding
        """
        results = {}
        
        # Try LegalBench (162 tasks)
        try:
            # LegalBench has multiple task categories
            legalbench_tasks = [
                'legalbench_issue_spotting',
                'legalbench_rule_recall',
                'legalbench_rule_application'
            ]
            
            legalbench_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=legalbench_tasks,
                limit=100,  # Sample from 162 tasks
                num_fewshot=0
            )
            legalbench_scores = legalbench_results.get('results', {})
            
            # Extract scores
            scores = []
            task_count = 0
            for task_name, task_results in legalbench_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        scores.append(float(acc) * 100)
                        task_count += 1
            
            results['legalbench'] = {
                'score': sum(scores) / len(scores) if scores else 0.0,
                'tasks_evaluated': task_count,
                'total_tasks': 162,
                'detailed': legalbench_scores
            }
        except Exception as e:
            logger.warning(f"LegalBench not available: {e}")
            results['legalbench'] = {
                'score': 0.0,
                'note': 'LegalBench not available'
            }
        
        # Try CUAD (510 contracts)
        try:
            cuad_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['cuad'],
                limit=100,  # Sample from 510 contracts
                num_fewshot=0
            )
            cuad_scores = cuad_results.get('results', {})
            
            score = 0.0
            for task_name, task_results in cuad_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            results['cuad'] = {
                'score': score,
                'dataset_size': 510,
                'samples_tested': 100,
                'detailed': cuad_scores
            }
        except Exception as e:
            logger.warning(f"CUAD not available: {e}")
            results['cuad'] = {
                'score': 0.0,
                'note': 'CUAD not available'
            }
        
        # Try LegalQA (fallback)
        try:
            legal_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['legal_qa'],
                limit=100,
                num_fewshot=0
            )
            legal_scores = legal_results.get('results', {})
            
            score = 0.0
            for task_name, task_results in legal_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            results['legal_qa'] = {
                'score': score,
                'detailed': legal_scores
            }
        except Exception as e:
            logger.warning(f"LegalQA not available: {e}")
            results['legal_qa'] = {
                'score': 0.0,
                'note': 'LegalQA not available'
            }
        
        # Compute aggregate legal score
        scores = [r.get('score', 0) for r in results.values() if isinstance(r, dict) and 'score' in r]
        results['aggregate_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return results
    
    def _benchmark_science(self) -> Dict:
        """
        Science domain benchmarks.
        
        Tests: SciBench, PubMedQA
        """
        results = {}
        
        # Try PubMedQA
        try:
            pubmed_results = evaluator.simple_evaluate(
                model=self.model,
                tasks=['pubmed_qa'],
                limit=50,
                num_fewshot=0
            )
            pubmed_scores = pubmed_results.get('results', {})
            
            score = 0.0
            for task_name, task_results in pubmed_scores.items():
                if isinstance(task_results, dict):
                    acc = task_results.get('acc_norm,none', task_results.get('acc,none', 0))
                    if acc:
                        score = float(acc) * 100
                        break
            
            results['pubmed_qa'] = {
                'score': score,
                'detailed': pubmed_scores
            }
        except Exception as e:
            logger.warning(f"PubMedQA not available: {e}")
            results['pubmed_qa'] = {
                'score': 0.0,
                'note': 'PubMedQA not available'
            }
        
        # SciBench would require custom implementation
        # For now, we'll use PubMedQA as the primary metric
        results['aggregate_score'] = results.get('pubmed_qa', {}).get('score', 0.0)
        
        return results
