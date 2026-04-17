# AGENTS.md

## Cursor Cloud specific instructions

This is a Python ML research project (no web server, no database, no Docker). All scripts run as standalone Python commands on CPU by default.

**Training Hub (draft product spec):** See [`docs/TRAINING_HUB_PRODUCT.md`](docs/TRAINING_HUB_PRODUCT.md) for a local-first training control plane concept (Colab + Vertex + Hugging Face publish, agent API parity, auto-research ecosystem). Not implemented in this repo unless/until added.

### Running the scripts

- **Graph Transformer training:** `python3 train.py --steps 500 --device cpu` (use fewer `--steps` for quick tests; the overfit sanity check runs first by default and takes ~30s on CPU)
- **Neuro-symbolic prototype:** `python3 prototypes/neuro_symbolic_vector_graph_prototype.py --epochs 1 --device cpu --no-progress` (requires internet on first run to download `distilbert-base-uncased` and `SemEvalWorkshop/sem_eval_2010_task_8` from Hugging Face Hub; add `--max-train-samples 200 --max-eval-samples 100` for a faster test)
- **KB graph viewer:** `python3 prototypes/view_kb_graph.py out/kb.json --out out/preview.png` (requires a prior prototype run that exported `out/kb.json`)
- **GUI inference preview (`infer_gui.py`):** requires Tkinter. On **macOS with Homebrew Python**, install the matching Tk add-on once (e.g. for 3.14): `brew install python-tk@3.14`, then verify with `python3 -m tkinter`. See comments at the top of `requirements.txt`.

### Caveats

- There are no automated test suites or linter configs in this repo. The overfit sanity check in `train.py` serves as the closest thing to an automated test.
- The neuro-symbolic prototype downloads ~250 MB of model weights on first run; subsequent runs use the Hugging Face cache.
- Use `--no-progress` flag on the prototype when running in non-TTY environments to avoid tqdm rendering issues.
- Matplotlib backend should be `Agg` (non-interactive) when running headless; `view_kb_graph.py` handles this automatically unless `--show` is passed.

### Local agent tools (optional)

- **Colab notebook CLI (community, not Google):** The PyPI package `colab-cli` provides the `colab-cli` command (push/pull/open `.ipynb` with Google Colab via PyDrive; needs OAuth `client_secrets.json` from Google Cloud Console — see `colab-cli set-config --help`). Install into an isolated venv: `python3 -m venv .venv-tools && .venv-tools/bin/pip install -r tools/requirements-tools.txt`, then run `.venv-tools/bin/colab-cli --help`. `GitPython` is listed explicitly because `colab-cli` imports `git` but does not always pull that dependency in cleanly.
- **Google Cloud / Colab Enterprise:** For official `gcloud colab` (runtimes, schedules, Enterprise), install the [Google Cloud SDK](https://cloud.google.com/sdk) and authenticate; it is separate from consumer colab.research.google.com notebooks.
- **Cursor Colab MCP:** The `colab-proxy-mcp` server exposes `open_colab_browser_connection` for browser-linked Colab sessions when that integration is enabled.
