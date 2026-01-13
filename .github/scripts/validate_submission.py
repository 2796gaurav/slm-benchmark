#!/usr/bin/env python3
"""
SLM Benchmark - Submission Validation Script
File: .github/scripts/validate_submission.py

Validates model submissions against requirements and checks for issues.
"""

import os
import sys
import yaml
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# HuggingFace API
from huggingface_hub import HfApi, model_info, list_repo_files


class SubmissionValidator:
    """Validates model submissions"""
    
    MAX_PARAMS = 3_000_000_000  # 3B parameters
    REQUIRED_FIELDS = [
        'name', 'family', 'hf_repo', 'parameters', 
        'architecture', 'license', 'submitted_by'
    ]
    VALID_LICENSES = [
        'Apache-2.0', 'MIT', 'BSD', 'CC-BY-4.0', 
        'CC-BY-SA-4.0', 'OpenRAIL', 'Llama 2', 'Llama 3'
    ]
    
    def __init__(self):
        self.hf_api = HfApi()
        self.errors = []
        self.warnings = []
    
    def validate_file(self, filepath: str) -> Tuple[bool, Dict]:
        """Main validation entry point"""
        self.errors = []
        self.warnings = []
        
        # Load YAML
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to parse YAML: {e}")
            return False, self._create_report(data=None)
        
        if 'model' not in data:
            self.errors.append("Missing 'model' key in YAML")
            return False, self._create_report(data)
        
        model = data['model']
        
        # Run all validations
        self._validate_required_fields(model)
        self._validate_parameters(model)
        self._validate_license(model)
        self._validate_hf_repo(model)
        self._validate_quantizations(model)
        self._validate_categories(model)
        self._validate_testing_config(model)
        self._check_submission_metadata(model)
        
        is_valid = len(self.errors) == 0
        report = self._create_report(model)
        
        return is_valid, report
    
    def _validate_required_fields(self, model: Dict):
        """Check all required fields are present"""
        for field in self.REQUIRED_FIELDS:
            if field not in model:
                self.errors.append(f"Missing required field: '{field}'")
            elif not model[field] or str(model[field]).strip() == '':
                self.errors.append(f"Field '{field}' cannot be empty")
    
    def _validate_parameters(self, model: Dict):
        """Validate parameter count is within limits"""
        if 'parameters' not in model:
            return
        
        param_str = str(model['parameters'])
        
        # Parse parameter string (e.g., "1.7B", "135M")
        try:
            num_params = self._parse_parameters(param_str)
            
            if num_params > self.MAX_PARAMS:
                self.errors.append(
                    f"Model has {param_str} parameters, exceeds 3B limit"
                )
            elif num_params < 1_000_000:  # 1M minimum
                self.warnings.append(
                    f"Model has only {param_str} parameters, very small for meaningful benchmarks"
                )
            
            model['_parsed_params'] = num_params
            
        except ValueError as e:
            self.errors.append(f"Invalid parameter format '{param_str}': {e}")
    
    def _parse_parameters(self, param_str: str) -> int:
        """Parse parameter string to integer"""
        import re
        
        match = re.match(r'^([\d.]+)([KMB])$', param_str.upper())
        if not match:
            raise ValueError("Format should be like '1.7B', '135M', '500K'")
        
        num = float(match.group(1))
        unit = match.group(2)
        
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
        return int(num * multipliers[unit])
    
    def _validate_license(self, model: Dict):
        """Validate license is acceptable"""
        if 'license' not in model:
            return
        
        license = model['license']
        
        # Check if license allows commercial use
        if license not in self.VALID_LICENSES:
            self.warnings.append(
                f"License '{license}' may have restrictions. Please verify it allows benchmarking."
            )
    
    def _validate_hf_repo(self, model: Dict):
        """Validate HuggingFace repository exists and is accessible"""
        if 'hf_repo' not in model:
            return
        
        repo_id = model['hf_repo']
        
        try:
            # Check if repo exists
            info = model_info(repo_id)
            
            # Verify it's a model repo
            if not info.pipeline_tag:
                self.warnings.append(
                    f"HuggingFace repo '{repo_id}' doesn't specify a pipeline_tag"
                )
            
            # Check for required files
            files = list_repo_files(repo_id)
            
            has_config = any('config.json' in f for f in files)
            has_model = any(f.endswith(('.safetensors', '.bin', '.gguf')) for f in files)
            
            if not has_config:
                self.warnings.append(f"No config.json found in repo '{repo_id}'")
            
            if not has_model:
                self.errors.append(
                    f"No model weights found in repo '{repo_id}'. "
                    "Ensure .safetensors or .bin files are present."
                )
            
            # Store repo info
            model['_repo_info'] = {
                'exists': True,
                'downloads': info.downloads,
                'likes': info.likes,
                'pipeline_tag': info.pipeline_tag
            }
            
        except Exception as e:
            self.errors.append(f"Cannot access HuggingFace repo '{repo_id}': {e}")
            model['_repo_info'] = {'exists': False}
    
    def _validate_quantizations(self, model: Dict):
        """Validate quantization configurations"""
        if 'quantizations' not in model:
            self.errors.append("No quantizations specified")
            return
        
        quants = model['quantizations']
        
        if not isinstance(quants, list) or len(quants) == 0:
            self.errors.append("Quantizations must be a non-empty list")
            return
        
        for idx, quant in enumerate(quants):
            if 'name' not in quant:
                self.errors.append(f"Quantization {idx} missing 'name' field")
            
            if 'format' not in quant:
                self.errors.append(f"Quantization {idx} missing 'format' field")
            
            # Validate format
            if 'format' in quant:
                valid_formats = ['safetensors', 'gguf', 'onnx']
                if quant['format'] not in valid_formats:
                    self.errors.append(
                        f"Quantization {idx} has invalid format '{quant['format']}'. "
                        f"Must be one of: {valid_formats}"
                    )
            
            # Check GGUF source if applicable
            if quant.get('format') == 'gguf' and 'source' not in quant:
                self.warnings.append(
                    f"Quantization {idx} (GGUF) should specify 'source' repo"
                )
    
    def _validate_categories(self, model: Dict):
        """Validate model categories"""
        if 'categories' not in model:
            self.warnings.append("No categories specified")
            return
        
        valid_categories = {
            'reasoning', 'coding', 'math', 'language',
            'multilingual', 'edge-optimized', 'instruction-following',
            'chat', 'function-calling', 'long-context'
        }
        
        categories = model['categories']
        
        if not isinstance(categories, list):
            self.errors.append("Categories must be a list")
            return
        
        for cat in categories:
            if cat not in valid_categories:
                self.warnings.append(
                    f"Unknown category '{cat}'. Valid categories: {valid_categories}"
                )
    
    def _validate_testing_config(self, model: Dict):
        """Validate testing configuration"""
        if 'testing' not in model:
            return  # Optional
        
        testing = model['testing']
        
        if 'priority' in testing:
            valid_priorities = ['standard', 'fast', 'comprehensive']
            if testing['priority'] not in valid_priorities:
                self.errors.append(
                    f"Invalid testing priority '{testing['priority']}'. "
                    f"Must be one of: {valid_priorities}"
                )
    
    def _check_submission_metadata(self, model: Dict):
        """Check submission metadata quality"""
        if 'submitted_by' in model:
            # Validate GitHub username format
            username = model['submitted_by']
            if not username.replace('-', '').replace('_', '').isalnum():
                self.warnings.append(
                    f"GitHub username '{username}' looks invalid"
                )
        
        if 'contact' in model:
            email = model['contact']
            if '@' not in email or '.' not in email:
                self.warnings.append(f"Email '{email}' looks invalid")
    
    def _create_report(self, model: Dict = None) -> Dict:
        """Create validation report"""
        report = {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': datetime.now().isoformat()
        }
        
        if model:
            report['model'] = {
                'name': model.get('name', 'Unknown'),
                'family': model.get('family', 'Unknown'),
                'parameters': model.get('parameters', 'Unknown'),
                'hf_repo': model.get('hf_repo', 'Unknown'),
                'license': model.get('license', 'Unknown'),
                'quantizations': len(model.get('quantizations', [])),
                'categories': model.get('categories', [])
            }
        
        return report


def check_duplicates(submission_file: str, registry_file: str) -> bool:
    """Check if model already exists in registry"""
    try:
        with open(submission_file, 'r') as f:
            submission = yaml.safe_load(f)
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        model_name = submission['model']['name']
        hf_repo = submission['model']['hf_repo']
        
        for existing in registry.get('models', []):
            if existing['name'] == model_name:
                print(f"❌ Model '{model_name}' already exists in registry")
                return True
            
            if existing['hf_repo'] == hf_repo:
                print(f"❌ HuggingFace repo '{hf_repo}' already benchmarked")
                return True
        
        return False
        
    except FileNotFoundError:
        # Registry doesn't exist yet (first submission)
        return False
    except Exception as e:
        print(f"⚠️  Warning: Could not check duplicates: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate SLM Benchmark submission')
    parser.add_argument('--submission-file', required=True, help='Path to submission YAML')
    parser.add_argument('--pr-number', type=int, help='Pull request number')
    parser.add_argument('--output', default='validation_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    # Validate submission
    validator = SubmissionValidator()
    is_valid, report = validator.validate_file(args.submission_file)
    
    # Check for duplicates if registry exists
    if is_valid and os.path.exists('models/registry.json'):
        is_duplicate = check_duplicates(args.submission_file, 'models/registry.json')
        if is_duplicate:
            report['valid'] = False
            report['errors'].append('Model already exists in registry')
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    if report['valid']:
        print("✅ Validation PASSED")
        print(f"\nModel: {report['model']['name']}")
        print(f"Parameters: {report['model']['parameters']}")
        print(f"HF Repo: {report['model']['hf_repo']}")
        print(f"Quantizations: {report['model']['quantizations']}")
    else:
        print("❌ Validation FAILED")
        print("\nErrors:")
        for error in report['errors']:
            print(f"  ❌ {error}")
    
    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings']:
            print(f"  ⚠️  {warning}")
    
    print("="*60 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if report['valid'] else 1)


if __name__ == '__main__':
    main()