# Training Hub — Product & Agent Specification

This document describes a **local-first control plane** for machine-learning training and research: one application that can target multiple execution backends (initially **Google Colab** and **Google Vertex AI**), unify run lifecycle and artifacts, optionally **publish to the Hugging Face Hub**, and expose **the same capabilities to human users and to automated agents** so the product can sit at the center of an **auto-research / auto-training ecosystem**.

It is a **product and architecture draft**, not a commitment to a particular tech stack. Implementation details can vary; **agent parity** and **auditability** are non-negotiable design constraints.

---

## 1. Problem statement

### 1.1 Fragmentation

- **Inference** has converged on recognizable hubs (local runners, API routers, chat UIs): picking a model and running it is relatively standardized.
- **Training** remains fragmented: notebooks (e.g. Colab), cloud ML platforms (e.g. Vertex), bare GPU rentals, and bespoke scripts each have different entrypoints, credentials, billing, and artifact conventions.
- **Researchers and small teams** repeatedly pay a **context-switching tax** (browser vs CLI vs Console vs `gcloud`) and an **automation tax** (hard to drive the same flows from scripts or agents without brittle screen automation).

### 1.2 Opportunity

A **local application** that:

- Presents a **single mental model** (projects, runs, artifacts, publish),
- Connects to **multiple backends** via explicit **adapters**,
- Exposes a **stable, versioned automation surface** (API + events) with **feature parity** for users and agents,

can become the **orchestration layer** for training—without replacing PyTorch, Hugging Face, or cloud vendors, and without pretending Colab and Vertex are one homogeneous API.

---

## 2. Product vision (one paragraph)

**Training Hub** is a local-first desktop or local web UI plus embedded automation server that lets users and agents **define training intent** (code, data references, hyperparameters, resource constraints), **execute** that intent on **Colab and/or Vertex** (and later other providers), **track** metrics and artifacts in a uniform way, and **publish** finished models (and optional datasets/cards) to **Hugging Face**—with **identical** programmatic control for trusted agents, enabling supervised **auto-research** and **auto-training** workflows.

---

## 3. Goals and non-goals

### 3.1 Goals

- **Unified run lifecycle** across backends: create, monitor, cancel, retrieve logs, list artifacts, finalize.
- **Unified artifact model**: checkpoints, configs, metrics summaries, provenance (git commit, dataset fingerprint, backend id).
- **Optional Hub publishing**: model repo creation/update, LFS-aware uploads, model card templates.
- **Agent parity**: anything a user can do in the UI must be available via a **documented, authenticated API** (within the same permission model).
- **Auditability**: who (user vs agent id) did what, when, on which backend, with what cost-relevant metadata where available.
- **Explicit backend differences**: the UI and API **name** the backend and surface limitations (e.g. Colab vs Vertex capabilities) instead of hiding failures behind vague errors.

### 3.2 Non-goals (initial phases)

- Replacing **Vertex** or **Colab** execution engines; this product **orchestrates** them, it does not reimplement managed training at hyperscale.
- Guaranteeing **single billing** or **single Google identity** across consumer Colab and GCP; users may still have **two** trust domains until vendors converge.
- Being the **authoritative experiment tracker** for every team (optional export to W&B, MLflow, etc., may come later).
- Unrestricted agent access by default: **agents are first-class but policy-scoped** (see §7).

---

## 4. Core concepts

These abstractions should appear in the UI, in persisted storage, and in the **agent API** with the same names and semantics.

| Concept | Definition |
|--------|------------|
| **Workspace** | Logical container for settings, secrets references (not plaintext), and quotas. May map 1:1 to “this machine” for local-first. |
| **Project** | User-facing grouping: repo path or archive, default configs, linked HF namespace preferences, tags. |
| **Run** | A single training (or evaluation) execution: immutable intent snapshot + mutable status + links to artifacts. |
| **Run intent** | Serializable spec: entrypoint (`train.py`, notebook, container image ref), environment hints, dataset references (HF dataset id, path, or URI), hyperparameters, resource request (GPU type/count, region, max runtime), backend choice. |
| **Backend** | A pluggable adapter: `colab`, `vertex`, later `local`, `ssh`, `runpod`, etc. |
| **Artifact** | File or logical object produced by a run: checkpoint shard, `config.yaml`, metrics JSON, logs tarball, eval report. Each has a **URI** (local cache path and/or cloud storage path) and **checksum** where applicable. |
| **Publication** | A structured action that creates/updates a **HF Model repo** (or Space later) from selected artifacts + metadata. |
| **Principal** | **User** (interactive) or **Agent** (automation): both authenticate to the Hub API; agents use separate credentials or tokens where supported. |

---

## 5. Backend adapters (initial)

### 5.1 Google Colab (consumer)

**Role:** Notebook-oriented and browser-tied execution; good for quick iteration and certain GPU access patterns.

**Reality:** There is **no** fully supported public “Colab API” equivalent to Vertex for all operations. An adapter must document **exactly** which operations are supported (e.g. open notebook, sync file to Drive, trigger execution via supported paths) and degrade gracefully.

**Adapter responsibilities:**

- Map **Run intent** to a **supported** Colab workflow (e.g. parameterized notebook, or script pushed to Drive + instructions).
- Surface **auth** requirements (OAuth for Drive/API where used); store tokens in OS secure storage.
- Return stable **external references** (file ids, notebook URLs) for correlation and debugging.

**Risk:** Brittleness. The product must **version** Colab adapter behavior and warn when Google changes behavior.

### 5.2 Vertex AI (GCP)

**Role:** First-class training jobs, pipelines, custom containers, enterprise IAM.

**Adapter responsibilities:**

- Map Run intent to **CustomJob** / **PipelineJob** / other chosen primitives (product decision per MVP).
- Use **Application Default Credentials** or **service account** keys **only** via secure storage and explicit user consent.
- Expose **job name, resource, region, billing-relevant labels** in the unified Run model.

**Benefit:** Stronger **automation** and **CLI-like** semantics than consumer Colab; better fit for agents running long jobs.

### 5.3 Hugging Face Hub (publish)

**Role:** Not a training backend; a **destination** for artifacts.

**Capabilities:**

- Create or update repos (model, and later dataset/space).
- Upload with **LFS** for large files; resumable uploads.
- Attach **model card** from template populated with Run metadata (training loss curves, eval, limitations, citation).

**Auth:** HF **user access tokens** with minimal scopes (write to repos the user owns or org permits).

---

## 6. Benefits summary

### 6.1 For end users

- **One app** to start and track training across Colab and Vertex.
- **Consistent artifact layout** locally (cache) and pointers to cloud storage.
- **Publish** to HF without learning Hub CLI details for the common path.
- **Clear separation** of “what I ran” vs “where it ran.”

### 6.2 For you (the product team)

- **Adapter pattern** lets you add providers without rewriting the core.
- **Agent API** creates differentiation vs “yet another dashboard.”
- **Run manifests** are a portable asset (export, sharing, templates).

### 6.3 For an auto-research / auto-training ecosystem

- Agents can **submit runs**, **poll status**, **consume metrics**, **decide next experiments**, and **publish** results under policy—**without** bespoke integrations per cloud UI.
- The same **audit trail** supports human review and compliance stories later.

---

## 7. Agent accessibility: full control (parity with users)

“Full control” means **feature parity** between interactive UI flows and the **automation API**, not “agents bypass safety.” Agents receive the **same capabilities** as the user **subject to the same policies and credentials**.

### 7.1 Principles

1. **API-first design**  
   Every UI action is implemented by calling internal services that are also exposed via **stable HTTP (localhost) or IPC** with a **versioned schema** (e.g. `/v1/...`).

2. **No hidden state**  
   Agents can query **workspace config**, **backend connectivity**, **run list**, **run detail**, and **artifact index** without scraping the UI.

3. **Idempotent operations where possible**  
   Create run with **client-supplied idempotency key**; publish with **content-addressed** artifact sets to avoid duplicate repos.

4. **Streaming and webhooks**  
   Long-running jobs expose **log streams** and **state transitions** via **Server-Sent Events or WebSocket**, plus **optional webhook callbacks** to user-controlled URLs for external orchestrators (with signing secrets).

5. **Structured errors**  
   Errors return **machine-readable codes** (`VERTEX_QUOTA`, `COLAB_AUTH_EXPIRED`, `HF_LFS_LIMIT`) and **remediation hints** for agents.

### 7.2 Authentication model for agents

| Mechanism | Purpose |
|-----------|---------|
| **Personal access token (PAT)** | Long-lived token minted in the Hub UI, scoped to **workspace + permissions** (see §7.4). Stored hashed server-side; shown once. |
| **mTLS or device-bound keys** (optional later) | Stronger guarantees for unattended machines. |
| **OAuth to the Hub itself** (optional) | For remote access beyond localhost; out of scope for strict local-first MVP. |

Human users authenticate to the **desktop app** normally. **Agents** authenticate to the **local API** with a PAT (or equivalent). **Cloud credentials** (GCP, HF, Google OAuth for Colab) remain **user-linked**: the agent never receives raw Google passwords; it uses tokens **the user has already authorized** for the Hub.

### 7.3 Permission matrix (example)

Define **roles** or **capability flags** attachable to each PAT:

| Capability | Description |
|------------|-------------|
| `runs:read` | List and inspect runs and artifacts. |
| `runs:write` | Create and cancel runs. |
| `publish:hf` | Create/update HF repos from artifacts. |
| `secrets:use` | Use stored backend credentials (never export raw secrets via API). |
| `secrets:manage` | Add/remove cloud credential bindings (highly restricted). |
| `admin:workspace` | Manage quotas, retention, agent tokens. |

**Default for a new agent token:** `runs:read`, `runs:write` on **one project**, no `secrets:manage`, optional `publish:hf` if explicitly enabled.

### 7.4 Parity checklist (acceptance criteria)

For each release, verify:

- [ ] Create run (all supported backends).
- [ ] Cancel run.
- [ ] Stream logs / poll status until terminal state.
- [ ] List and download artifacts (local cache + signed URLs if cloud-backed).
- [ ] Re-run from prior **Run intent** (clone spec).
- [ ] Publish to HF (dry-run mode optional: validate token and repo name without upload).
- [ ] Enumerate backends and their **capability flags** (`supports_distributed`, `max_runtime`, etc.).
- [ ] All of the above via **API** with identical outcomes to UI (automated tests).

### 7.5 MCP and tool ecosystem

To integrate with **Cursor** and similar hosts:

- Provide an **MCP server** implementation that maps **tool calls** to the same **internal service layer** as HTTP (not a second implementation).
- Tools should be **coarse-grained** where possible (`submit_run`, `get_run_status`) to reduce round-trips, with **fine-grained** tools optional (`upload_file_to_run`).

Document **rate limits** and **recommended polling intervals** so agents do not hammer Vertex/Colab APIs.

### 7.6 Safety and abuse controls (agents)

- **Spend caps** per workspace/day/backend (where measurable).
- **Confirmation policy**: optional **human-in-the-loop** gate for `runs:write` above a threshold or for `publish:hf`.
- **Immutable audit log**: principal id, PAT id (hashed), action, params summary, timestamp, outcome.
- **Kill switch**: revoke all agent tokens for workspace in one action.

---

## 8. Auto-research / auto-training ecosystem

### 8.1 What this enables

With parity and policies:

1. **Planner agents** propose the next hyperparameters or data subset based on prior **Run** results read from the API.
2. **Executor agents** submit jobs and wait on **streaming metrics**.
3. **Curator agents** assemble model cards and **publish** to HF when metrics meet gates.
4. **Reviewers** (humans or rules) inspect the **audit log** and artifact lineage before widening access.

### 8.2 Minimum data for closed-loop automation

The Hub should persist and expose:

- **Run intent** (full JSON).
- **Metric time series** (or pointers to TensorBoard/W&B export if integrated later).
- **Eval summaries** (structured JSON).
- **Git revision** and **dirty/clean** flag when run from a repo.
- **Dataset identity** (HF revision, split hash, or file manifest hash).

### 8.3 Optional future hooks

- **Webhook** on `run.completed` with signed payload for external schedulers.
- **Plugin SDK** for custom **stopping criteria** or **hyperband** logic running **inside** the Hub process (trusted code only).

---

## 9. Suggested technical shape (non-binding)

- **Local process**: embedded API server (localhost-only by default), single workspace database (SQLite or embedded store) for runs and metadata; large blobs on disk or symlinked to object storage.
- **Schema**: JSON Schema or Protobuf for **Run intent** and **API**; version all payloads.
- **Secrets**: OS keychain integration for GCP SA JSON paths, HF tokens, OAuth refresh tokens.
- **Colab adapter**: isolated module with explicit **capability manifest** and **simulation mode** for tests.

---

## 10. Roadmap (illustrative)

| Phase | Scope |
|-------|--------|
| **MVP** | Vertex adapter + local artifact store + PAT API + minimal UI; HF publish for single checkpoint + card. |
| **Phase 2** | Colab adapter with documented limitations + PAT scopes + MCP server. |
| **Phase 3** | Webhooks, spend caps, multi-project, export to MLflow/W&B. |
| **Phase 4** | Additional compute adapters (SSH, RunPod, …) behind same Run model. |

---

## 11. Open questions

- Which **Vertex** primitive is the default mapping for “Run” (CustomJob vs Pipeline vs Notebook runner)?
- How to represent **notebook-first** Colab workflows in a **script-first** Run model without lying to users.
- **Retention**: how long to keep local caches and whether to enforce **artifact GC** policies.
- **Multi-user** vs single-user local assumption for v1.

---

## 12. Document control

- **Owner:** product / engineering (TBD).
- **Audience:** internal planning, investors, and engineering onboarding.
- **Revision:** bump version note when Colab/Vertex integration assumptions change.

**Version:** 0.1 (draft)
