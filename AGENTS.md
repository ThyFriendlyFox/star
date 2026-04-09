# AGENTS.md

## Cursor Cloud specific instructions

This is a Python ML research project (no web server, no database, no Docker). All scripts run as standalone Python commands on CPU by default.

### Running the scripts

- **Graph Transformer training:** `python3 train.py --steps 500 --device cpu` (use fewer `--steps` for quick tests; the overfit sanity check runs first by default and takes ~30s on CPU)
- **Neuro-symbolic prototype:** `python3 prototypes/neuro_symbolic_vector_graph_prototype.py --epochs 1 --device cpu --no-progress` (requires internet on first run to download `distilbert-base-uncased` and `SemEvalWorkshop/sem_eval_2010_task_8` from Hugging Face Hub; add `--max-train-samples 200 --max-eval-samples 100` for a faster test)
- **KB graph viewer:** `python3 prototypes/view_kb_graph.py out/kb.json --out out/preview.png` (requires a prior prototype run that exported `out/kb.json`)

### Caveats

- There are no automated test suites or linter configs in this repo. The overfit sanity check in `train.py` serves as the closest thing to an automated test.
- The neuro-symbolic prototype downloads ~250 MB of model weights on first run; subsequent runs use the Hugging Face cache.
- Use `--no-progress` flag on the prototype when running in non-TTY environments to avoid tqdm rendering issues.
- Matplotlib backend should be `Agg` (non-interactive) when running headless; `view_kb_graph.py` handles this automatically unless `--show` is passed.
