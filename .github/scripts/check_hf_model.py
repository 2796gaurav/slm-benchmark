import argparse
import sys
from huggingface_hub import HfApi, hf_hub_url
import yaml
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-file", required=True)
    args = parser.parse_args()

    files = glob.glob(args.submission_file)
    if not files:
        print("No submission file found.")
        sys.exit(1)

    with open(files[0], 'r') as f:
        config = yaml.safe_load(f)

    model_id = config.get('model', {}).get('hf_repo')
    if not model_id:
        print("No HF repo specified in submission.")
        sys.exit(1)

    api = HfApi()
    try:
        api.model_info(model_id)
        print(f"Model {model_id} exists on Hugging Face.")
    except Exception as e:
        print(f"Error: Model {model_id} not found or inaccessible. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
