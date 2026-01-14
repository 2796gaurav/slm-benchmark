## Contributing to SLM Benchmark

Thank you for considering a contribution to **SLM Benchmark**. This project aims to be a practical, transparent
benchmark for **Small Language Models (SLMs, ~1M–3B parameters)** that runs reliably on CPU (including GitHub Actions).

### Ways to Contribute

- **Submit a model**: Add a new SLM to the benchmark.
- **Improve evaluation**: Suggest or implement new tasks for small models.
- **Website & UX**: Polish the leaderboard and model pages.
- **Docs & Tutorials**: Improve the README, guides, or troubleshooting.

### Development Workflow

1. **Fork & clone**
   ```bash
   git clone https://github.com/2796gaurav/slm-benchmark.git
   cd slm-benchmark
   ```
2. **Create a branch**
   ```bash
   git checkout -b feature/my-change
   ```
3. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install pytest
   ```
4. **Run tests**
   ```bash
   pytest
   ```
5. **Open a Pull Request**
   - Keep PRs focused and small where possible.
   - Add/update tests for behavior changes.
   - Update documentation when behavior or interfaces change.

### Model Submissions

- Use the template in `models/submissions/template.yml`.
- Keep model size ≤ 3B parameters.
- The model must be:
  - Public on Hugging Face.
  - Licensed to allow benchmarking and result publication.

See the **“Submission Guide”** section in `README.md` for details.

### Code Style & Expectations

- Python 3.11+.
- Prefer type hints and docstrings for new public functions.
- Avoid adding GPU‑only dependencies or assumptions; all core paths must work on CPU.
- Do not add TODOs or commented‑out dead code—keep the codebase clean.

### Contact

- Maintainer: **@2796gaurav** on GitHub.
- For questions, please open a GitHub Issue or Discussion instead of contacting privately where possible.


