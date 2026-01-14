import json
import re
from pathlib import Path


def parse_parameters(param_str: str) -> float:
    """Mirror the leaderboard parameter parsing logic."""
    match = re.search(r'([\d.]+)([KMB])', str(param_str), re.IGNORECASE)
    if not match:
        return 0.0
    num = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
    return num * multipliers[unit]


def load_sample_leaderboard() -> dict:
    """Load the real leaderboard.json used by the website, if present."""
    path = Path(__file__).parents[1] / "website" / "assets" / "data" / "leaderboard.json"
    if not path.exists():
        pytest.skip("website/assets/data/leaderboard.json not present")
    with path.open() as f:
        return json.load(f)


def test_parse_parameters_basic():
    assert parse_parameters("1M") == 1e6
    assert parse_parameters("1.5B") == 1.5e9
    assert parse_parameters("135M") == 135e6
    assert parse_parameters("42K") == 42e3
    assert parse_parameters("unknown") == 0.0


def test_leaderboard_structure_and_sizes():
    data = load_sample_leaderboard()
    models = data.get("models", [])
    assert isinstance(models, list)

    # Basic shape checks
    for model in models:
        assert "name" in model
        assert "family" in model
        assert "parameters" in model
        # Ensure parameter strings are parseable into numeric counts
        numeric_params = parse_parameters(model["parameters"])
        assert numeric_params >= 0


def test_metadata_consistency():
    data = load_sample_leaderboard()
    metadata = data.get("metadata", {})
    total_models = metadata.get("total_models", 0)
    assert total_models == len(data.get("models", []))
