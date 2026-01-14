#!/usr/bin/env python3
"""
Long Context Evaluation
Implements Needle-in-Haystack and multi-document QA tests
"""

import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LongContextEvaluator:
    """
    Evaluates model performance on long context tasks.
    
    Implements:
    - Needle in Haystack: Finding specific information in long documents
    - Multi-document QA: Answering questions across multiple documents
    - Long-form coherence: Maintaining coherence in long generations
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.max_context_length = getattr(config, 'context_length', 8192)
    
    def evaluate_all(self) -> Dict:
        """Run all long context evaluations."""
        try:
            results = {
                'needle_in_haystack': self.needle_in_haystack(),
                'multi_doc_qa': self.multi_document_qa(),
                'long_form_coherence': self.long_form_coherence()
            }
            return results
        except Exception as e:
            logger.error(f"Error in long context evaluation: {e}")
            return {
                'needle_in_haystack': 0.0,
                'multi_doc_qa': 0.0,
                'long_form_coherence': 0.0,
                'error': str(e)
            }
    
    def needle_in_haystack(self, num_tests: int = 10) -> float:
        """
        Needle in Haystack test: Place a specific fact in a long document
        and test if the model can retrieve it.
        
        Args:
            num_tests: Number of test cases to run
            
        Returns:
            Accuracy score (0-100)
        """
        logger.info("Running Needle in Haystack evaluation")
        
        # Sample needles (facts to find)
        needles = [
            ("The magic number is 7294", "What is the magic number?", "7294"),
            ("The secret code is ALPHA-BRAVO-42", "What is the secret code?", "ALPHA-BRAVO-42"),
            ("The treasure is hidden in the old oak tree", "Where is the treasure hidden?", "old oak tree"),
            ("The meeting is scheduled for 3:45 PM", "When is the meeting?", "3:45 PM"),
            ("The password is BlueSky2026", "What is the password?", "BlueSky2026"),
        ]
        
        correct = 0
        total = 0
        
        for needle, question, answer in needles[:num_tests]:
            try:
                # Create haystack (filler text)
                haystack = self._generate_haystack(target_length=2000)
                
                # Insert needle at random position
                insert_pos = random.randint(0, len(haystack))
                full_text = haystack[:insert_pos] + " " + needle + " " + haystack[insert_pos:]
                
                # Truncate if too long
                max_chars = self.max_context_length * 4  # Rough estimate: 4 chars per token
                if len(full_text) > max_chars:
                    full_text = full_text[:max_chars]
                
                # Ask question
                prompt = f"{full_text}\n\nQuestion: {question}\nAnswer:"
                
                try:
                    response = self.model.generate(prompt, max_tokens=50)
                    
                    # Check if answer is in response
                    if answer.lower() in response.lower():
                        correct += 1
                    
                    total += 1
                except Exception as e:
                    logger.warning(f"Error generating response: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error in needle test: {e}")
                continue
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        logger.info(f"Needle in Haystack: {correct}/{total} correct ({accuracy:.1f}%)")
        return accuracy
    
    def multi_document_qa(self) -> float:
        """
        Multi-document QA: Answer questions that require information
        from multiple documents.
        
        Returns:
            Accuracy score (0-100)
        """
        logger.info("Running multi-document QA evaluation")
        
        # Simplified test cases
        test_cases = [
            {
                'docs': [
                    "Document 1: The company was founded in 2010.",
                    "Document 2: The company has 500 employees.",
                    "Document 3: The company's revenue is $50 million."
                ],
                'question': "How many employees does the company founded in 2010 have?",
                'answer': "500"
            },
            {
                'docs': [
                    "Article A: Paris is the capital of France.",
                    "Article B: The Eiffel Tower is 330 meters tall.",
                    "Article C: The Eiffel Tower was completed in 1889."
                ],
                'question': "How tall is the famous tower in France's capital?",
                'answer': "330"
            }
        ]
        
        correct = 0
        total = 0
        
        for test in test_cases:
            try:
                # Combine documents
                combined = "\n\n".join(test['docs'])
                prompt = f"{combined}\n\nQuestion: {test['question']}\nAnswer:"
                
                response = self.model.generate(prompt, max_tokens=50)
                
                if test['answer'].lower() in response.lower():
                    correct += 1
                
                total += 1
            except Exception as e:
                logger.warning(f"Error in multi-doc QA: {e}")
                continue
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        logger.info(f"Multi-document QA: {correct}/{total} correct ({accuracy:.1f}%)")
        return accuracy
    
    def long_form_coherence(self) -> float:
        """
        Test coherence in long-form generation.
        
        Returns:
            Coherence score (0-100) based on simple heuristics
        """
        logger.info("Running long-form coherence evaluation")
        
        prompts = [
            "Write a detailed explanation of how photosynthesis works:",
            "Describe the history of the internet from its inception to today:",
            "Explain the process of machine learning in detail:"
        ]
        
        coherence_scores = []
        
        for prompt in prompts:
            try:
                response = self.model.generate(prompt, max_tokens=500)
                
                # Simple coherence heuristics
                score = self._assess_coherence(response)
                coherence_scores.append(score)
                
            except Exception as e:
                logger.warning(f"Error in coherence test: {e}")
                continue
        
        avg_score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        logger.info(f"Long-form coherence: {avg_score:.1f}")
        return avg_score
    
    def _generate_haystack(self, target_length: int = 2000) -> str:
        """Generate filler text for haystack."""
        filler_sentences = [
            "The sun was shining brightly in the clear blue sky.",
            "Birds were singing in the trees nearby.",
            "A gentle breeze rustled through the leaves.",
            "The river flowed peacefully through the valley.",
            "Mountains stood tall in the distance.",
            "Clouds drifted slowly across the horizon.",
            "The forest was full of life and activity.",
            "Flowers bloomed in vibrant colors everywhere.",
        ]
        
        haystack = []
        current_length = 0
        
        while current_length < target_length:
            sentence = random.choice(filler_sentences)
            haystack.append(sentence)
            current_length += len(sentence)
        
        return " ".join(haystack)
    
    def _assess_coherence(self, text: str) -> float:
        """
        Simple coherence assessment based on heuristics.
        
        In production, use more sophisticated metrics like:
        - Perplexity
        - Semantic similarity between sentences
        - Topic consistency
        """
        if not text or len(text) < 50:
            return 0.0
        
        score = 50.0  # Base score
        
        # Check for repetition (bad)
        sentences = text.split('.')
        unique_sentences = set(s.strip() for s in sentences if s.strip())
        if len(sentences) > 0:
            repetition_ratio = len(unique_sentences) / len(sentences)
            score += repetition_ratio * 25
        
        # Check for reasonable length
        if 100 < len(text) < 2000:
            score += 15
        
        # Check for sentence structure
        if text.count('.') > 2:  # Multiple sentences
            score += 10
        
        return min(score, 100.0)
