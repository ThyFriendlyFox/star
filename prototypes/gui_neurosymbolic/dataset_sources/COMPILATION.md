# Public GUI / computer-use datasets — compilation notes

This file summarizes how four public sources relate to the `gui_neurosymbolic` prototype and the canonical batch keys described in `public_gui_schema.py` (`image`, `task_token_ids`, `task_text`, action heads, etc.). None of these sets ship in the same tensor schema as this repo’s synthetic loader; each needs an adapter that maps **screens + task text + (optional) action traces** into `CanonicalSample`.

Set `HF_TOKEN` when hitting the Hugging Face Hub to avoid rate limits.

---

## 1. [arthurcolle/open-computer-using-agent](https://huggingface.co/datasets/arthurcolle/open-computer-using-agent)

| | |
|---|---|
| **License** | MIT (per Hub card) |
| **Rough size** | ~686 MB total download (Hub) |
| **Primary modality** | Chat-style `messages` plus (per card) cursor, windows, display metadata in collection |

**Hub viewer / schema issues:** The dataset viewer can fail with Arrow/JSON errors because `messages[].content` is not a single type across rows (string vs structured content in some lines). That does not necessarily block client-side loading.

**Loading:** `datasets.load_dataset(..., split="train", streaming=True)` works in practice; the first streamed row may only expose `messages` (list of `{role, content, name}`), with string `content` in the sample we tried.

**Mapping:** Treat as **instruction + (optional) screenshots encoded in messages or sidecar fields** once you normalize `content` to text. You will need a custom parser if `content` is sometimes a list of segments (multimodal chat).

---

## 2. [xlangai/AgentNet](https://huggingface.co/datasets/xlangai/AgentNet)

| | |
|---|---|
| **License** | MIT |
| **Paper** | [arXiv:2508.09123](https://arxiv.org/abs/2508.09123) |

**Hub viewer / schema issues:** Full materialization can fail with `DatasetGenerationCastError` because `meta_data_merged.jsonl` mixes **two different column sets** (metadata-only rows vs rows with `traj`, `task_completed`, etc.). The Hub error message lists the mismatched columns explicitly.

**Loading strategies:**

- **Streaming** (`streaming=True`): Often succeeds for the first rows and exposes a consistent schema *for that stream segment* (e.g. `task_id`, `instruction`, `traj`, `domain`, `task_completed`, …). This is suitable for exploration, not a guarantee that all rows match.
- **Robust training:** Split the JSONL yourself (two passes or filter by presence of `traj`), or ask upstream for separate files / configs per schema.

**Mapping:** `instruction` / `natural_language_task` → `task_text`. `traj` entries reference step images (e.g. PNG filenames) and code strings — resolve files against the dataset’s image storage layout, then stack or sample frames for `image`.

---

## 3. [anaisleila/computer-use-data-psai](https://huggingface.co/datasets/anaisleila/computer-use-data-psai) (from [computeruse-data-psai](https://github.com/anaishowland/computeruse-data-psai))

| | |
|---|---|
| **License** | MIT |
| **Scale** | ~3.1k unique tasks; **~7.87 GB** parquet with embedded screenshots; **~49 GB** if you clone videos + DOM artifacts |

The GitHub repo is documentation and examples; the **canonical Hub id** is `anaisleila/computer-use-data-psai`.

**Loading:**

```python
from datasets import load_dataset
ds = load_dataset("anaisleila/computer-use-data-psai")
```

Per the upstream README: **100 duplicate rows** — deduplicate on `unique_data_id` before training. Videos and DOM zips are referenced by path fields and are meant to be fetched on demand (`huggingface_hub.hf_hub_download`), not necessarily all at once.

**Mapping:** Rich fields for GUI neuro-symbolic work: `task_name`, `screenshots` (embedded images), `events` (JSON string), `reasoning_steps`, `metadata` (JSON string). Map `task_name` → `task_text`; choose one or more `screenshots` → `image`; parse `events` for action supervision if you extend beyond the prototype’s grid head.

**Training in this repo:** `prototypes/gui_neurosymbolic/train.py --data-source psai` streams the train split (only the columns needed for images + events), buffers `--train-samples` + `--eval-samples` rows in one pass, derives `click_cell` / bbox bins from the first `click` in `events` (normalized by `screen_width` / `screen_height` in `metadata`), and hashes text for auxiliary heads. Requires Hugging Face access; set `HF_TOKEN` for reliable throughput.

---

## 4. [henryhe0123/PC-Agent-E](https://huggingface.co/datasets/henryhe0123/PC-Agent-E)

| | |
|---|---|
| **License** | MIT |
| **Paper** | [arXiv:2505.13909](https://arxiv.org/abs/2505.13909) |
| **Hub layout** | `data.zip` at repo root (thousands of files under `data/events/`) |

**Critical Hub quirk (verified locally):** The auto-built `train` split advertises an `Image` column, but Arrow rows can have **`bytes: null`** and **`path`** pointing at a **`zip://.../data.zip/.../*.jsonl`** URL — not an image file. `datasets` then raises `PIL.UnidentifiedImageError` when decoding. So **`load_dataset` is not a reliable image source** for this revision until the Hub metadata is fixed.

**Practical workaround:** Download `data.zip` with `hf_hub_download`, then pair:

- `data/events/screenshot/*.png` (valid PNG magic bytes in samples), and  
- matching `data/events/task*.jsonl` / `.md` traces under `data/events/`

Build your own manifest (task id → screenshot paths + instruction text from events).

**Split verification:** Loading may report `NonMatchingSplitsSizesError` between card metadata and generated shards; `verification_mode="no_checks"` can silence that — it does **not** fix the image path issue above.

**Mapping:** After you build paths from the zip, feed PNGs into `image` and text from JSONL into `task_text` / tokenizer.

---

## Quick comparison

| Dataset | Stable `datasets` image decode | Action / trace signal | Main friction |
|---------|-------------------------------|------------------------|-----------------|
| open-computer-using-agent | N/A (chat-first) | Implicit in messages / state | Inconsistent `content` typing |
| AgentNet | Via `traj` image refs | Strong (`traj`, code) | Mixed JSONL schemas in one file |
| PSAI | Yes (embedded screenshots) | `events` JSON, reasoning | Size + optional large artifacts |
| PC-Agent-E | **Broken on Hub** for images | In jsonl under zip | Use zip directly; ignore broken Image column |

---

## Script

`compile_hf_datasets.py` (same directory) probes these ids and prints keys or errors without training the model.
