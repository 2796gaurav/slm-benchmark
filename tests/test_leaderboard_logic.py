import json

# Simulated leaderboard.json
data = {
  "models": [
    {
      "id": "tinystories-1m-verify",
      "name": "TinyStories-1M-Verify",
      "family": "TinyStories",
      "hf_repo": "roneneldan/TinyStories-1M",
      "parameters": "1M",
      "license": "Apache-2.0",
      "aggregate_score": 0.0,
      "scores": {
        "reasoning": 0.0,
        "coding": 0.0,
        "math": 0.0,
        "language": 0.0,
        "edge": 0.0,
        "safety": 0.0
      },
      "quantizations": [
        {
          "name": "FP32",
          "format": "transformers"
        }
      ],
      "categories": [
        "language"
      ],
      "date_added": "2026-01-13",
      "submitted_by": "2796gaurav",
      "rank": 1
    }
  ],
  "metadata": {
    "last_updated": "2026-01-13T15:27:54.253547",
    "total_models": 1
  }
}

def parse_parameters(param_str):
    import re
    match = re.search(r'([\d.]+)([KMB])', param_str, re.IGNORECASE)
    if not match: return 0
    num = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = { 'K': 1e3, 'M': 1e6, 'B': 1e9 }
    return num * multipliers[unit]

# Test logic
models = data.get('models', [])
print(f"Total models: {len(models)}")
for m in models:
    name = m['name']
    params = parse_parameters(m['parameters'])
    score = m['aggregate_score']
    print(f"Model: {name}, Params: {params}, Score: {score}")
    
    # Check if filters work
    search = ""
    matches_search = search.lower() in m['name'].lower() or search.lower() in m['family'].lower() or search.lower() in m['hf_repo'].lower()
    print(f"Matches search '': {matches_search}")
    
    size_filter = "all"
    is_all = size_filter == "all"
    print(f"Matches size 'all': {is_all}")
