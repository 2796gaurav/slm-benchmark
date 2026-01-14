import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelAutoDetector:
    """
    Automatically detects ALL model properties from HuggingFace repo
    """
    def __init__(self):
        self.hf_api = HfApi()
    
    def detect_all_properties(self, hf_repo: str) -> Dict[str, Any]:
        """
        Download config.json and detect everything automatically
        """
        logger.info(f"Detecting properties for {hf_repo}...")
        try:
            # Download config.json from repo
            config_path = hf_hub_download(
                repo_id=hf_repo,
                filename="config.json"
            )
            
            with open(config_path) as f:
                config = json.load(f)
            
            # Auto-detect all fields
            detected = {
                'name': self._extract_name(hf_repo),
                'parameters': self._calculate_params(config),
                'architecture': config.get('architectures', ['unknown'])[0],
                'context_length': self._get_context_length(config),
                'quantizations': self._detect_quant_formats(hf_repo),
                'vocab_size': config.get('vocab_size'),
                'hidden_size': config.get('hidden_size'),
                'num_layers': config.get('num_hidden_layers'),
                'hf_repo': hf_repo
            }
            
            logger.info(f"Successfully detected properties: {json.dumps(detected, indent=2)}")
            return detected
            
        except RepositoryNotFoundError:
            logger.error(f"Repository {hf_repo} not found on HuggingFace Hub")
            raise ValueError(f"Repository {hf_repo} not found")
        except Exception as e:
            logger.error(f"Error detecting properties: {str(e)}")
            raise ValueError(f"Cannot access repo {hf_repo}: {str(e)}")
    
    def _extract_name(self, hf_repo: str) -> str:
        """Extract a readable name from the repo ID"""
        return hf_repo.split('/')[-1]

    def _calculate_params(self, config: Dict) -> str:
        """
        Estimate parameters from architecture config or use actual param count if available
        """
        # Try to calculate from config dimensions if not explicitly stated
        hidden = config.get('hidden_size', 0)
        layers = config.get('num_hidden_layers', 0)
        vocab = config.get('vocab_size', 0)
        
        # Rough estimation formula: 
        # 12 * layers * hidden^2 + vocab * hidden
        # This is a very rough approximation for Transformer-based models
        params = (12 * layers * (hidden ** 2)) + (vocab * hidden)
        
        # If parameters seem too low (config missing), try to fallback or just return 'Unknown'
        if params == 0:
             return "Unknown"

        # Convert to human readable
        if params > 1e9:
            return f"{params/1e9:.1f}B"
        elif params > 1e6:
            return f"{params/1e6:.1f}M"
        return str(params)
    
    def _detect_quant_formats(self, hf_repo: str) -> List[str]:
        """
        Scan repo files to find available quantization formats
        """
        try:
            files = self.hf_api.list_repo_files(hf_repo)
            formats = []
            
            if any(f.endswith('.safetensors') for f in files) or any(f.endswith('.bin') for f in files):
                formats.append('FP16')
            if any('gguf' in f.lower() for f in files):
                formats.append('GGUF')
            if any('int8' in f.lower() for f in files):
                formats.append('INT8')
            if any('int4' in f.lower() or 'awq' in f.lower() or 'gptq' in f.lower() for f in files):
                formats.append('INT4')
                
            return formats or ['FP16']  # Default to FP16 if nothing else specific found
        except Exception as e:
            logger.warning(f"Could not list repo files: {e}")
            return ['FP16']
    
    def _get_context_length(self, config: Dict) -> int:
        """Extract context window size"""
        return config.get('max_position_embeddings', 
                         config.get('n_positions', 2048))

def main():
    parser = argparse.ArgumentParser(description='Auto-detect model properties from HuggingFace')
    parser.add_argument('--hf-repo', type=str, required=False, help='HuggingFace Repository ID')
    parser.add_argument('--yaml-file', type=str, required=False, help='Path to submission YAML file to extract repo from')
    parser.add_argument('--output', type=str, default='detected.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    repo_id = args.hf_repo
    
    if not repo_id and args.yaml_file:
        try:
            import yaml
            with open(args.yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                repo_id = data.get('model', {}).get('hf_repo')
        except Exception as e:
            logger.error(f"Failed to read YAML file: {e}")
            sys.exit(1)
            
    if not repo_id:
        logger.error("Must provide either --hf-repo or --yaml-file")
        sys.exit(1)
        
    detector = ModelAutoDetector()
    try:
        props = detector.detect_all_properties(repo_id)
        
        with open(args.output, 'w') as f:
            json.dump(props, f, indent=2)
            
        print(f"Successfully saved detected properties to {args.output}")
        
    except Exception as e:
        logger.error(f"Failed to detect properties: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
