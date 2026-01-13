# scripts/validate_model.py
"""
Validates model configuration files before benchmarking
"""
import sys
import yaml
import requests
from pathlib import Path

MAX_PARAMS = 3_000_000_000
ALLOWED_LICENSES = ['apache-2.0', 'mit', 'llama', 'gemma', 'cc-by-4.0']

def validate_model_config(config_path: Path) -> bool:
    """Validate a model configuration file"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['name', 'hf_model_id', 'parameters', 'license']
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Validate parameter count
    if config['parameters'] > MAX_PARAMS:
        print(f"❌ Model too large: {config['parameters']:,} params (max: {MAX_PARAMS:,})")
        return False
    
    # Validate HuggingFace model exists
    model_id = config['hf_model_id']
    response = requests.get(f"https://huggingface.co/api/models/{model_id}")
    if response.status_code != 200:
        print(f"❌ Model not found on HuggingFace: {model_id}")
        return False
    
    # Validate license
    license_lower = config['license'].lower()
    if not any(allowed in license_lower for allowed in ALLOWED_LICENSES):
        print(f"⚠️  Warning: Unusual license '{config['license']}'")
    
    # Validate quantization configs
    if 'quantization' in config:
        for quant_config in config['quantization']:
            if 'type' not in quant_config:
                print(f"❌ Missing 'type' in quantization config")
                return False
            
            if quant_config['type'] not in ['none', 'gptq', 'awq', 'gguf']:
                print(f"❌ Unknown quantization type: {quant_config['type']}")
                return False
    
    print(f"✅ Model config valid: {config['name']}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_model.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"❌ File not found: {config_path}")
        sys.exit(1)
    
    is_valid = validate_model_config(config_path)
    sys.exit(0 if is_valid else 1)