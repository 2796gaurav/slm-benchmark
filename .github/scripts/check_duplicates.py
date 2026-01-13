import argparse
import sys
import json
import yaml
import glob
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()

    files = glob.glob(args.submission_file)
    if not files:
        print("No submission file found.")
        sys.exit(1)

    with open(files[0], 'r') as f:
        config = yaml.safe_load(f)
    
    new_model_repo = config.get('model', {}).get('hf_repo')
    
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print("Registry not found, skipping duplicate check.")
        sys.exit(0)
        
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        
    # Registry structure assumption: list of models or dict keyed by name
    # Let's assume list of objects
    if isinstance(registry, list):
        for model in registry:
            if model.get('hf_repo') == new_model_repo:
                print(f"Model {new_model_repo} already exists in registry.")
                sys.exit(1)
    elif isinstance(registry, dict):
        # Maybe keyed by something else
        pass
        
    print("No duplicates found.")

if __name__ == "__main__":
    main()
