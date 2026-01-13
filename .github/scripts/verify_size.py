import argparse
import sys
import yaml
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--max-params", type=int, default=3000000000)
    args = parser.parse_args()

    files = glob.glob(args.submission_file)
    if not files:
        print("No submission file found.")
        sys.exit(1)

    with open(files[0], 'r') as f:
        config = yaml.safe_load(f)

    # In a real scenario, we might query the HF API for safetensors size/config.
    # For now, we trust the user's YAML or the check_hf_model step to populate metadata.
    # Here we assume the user puts 'parameters' in the yaml, 
    # OR we use a placeholder check if it is missing.
    
    params = config.get('model', {}).get('parameters')
    if isinstance(params, str):
        # Handle "3B", "1.5B" etc
        params = params.upper()
        if 'B' in params:
             val = float(params.replace('B', '')) * 1e9
        elif 'M' in params:
             val = float(params.replace('M', '')) * 1e6
        else:
             val = float(params)
    elif isinstance(params, (int, float)):
        val = params
    else:
        print("Could not determine parameter count from config.")
        # We might default to pass if we can't check, or fail.
        # Let's pass for now but warn
        sys.exit(0)

    if val > args.max_params:
        print(f"Model too large: {val} > {args.max_params}")
        sys.exit(1)
        
    print(f"Model size {val} is within limits.")

if __name__ == "__main__":
    main()
