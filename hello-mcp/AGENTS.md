# Agent Guidelines for the `hello-mcp` Workspace

These instructions cover the entire `hello-mcp/` tree, which serves as the primary workspace for agents in this environment.

## Quick Onboarding Walkthrough (10–15 min)
1. **GPU & environment check**
   - Call `gpu_info()` → confirm an available A100 (or note otherwise).
   - `job_list(limit=50)` → skim recent runs; pick the latest active/failed job IDs.
   - `job_logs(job_id, tail_lines=200, squash_repeats=true)` → understand why a run stopped.
2. **Read me first (local docs & layout)**
   - Open `AGENTS.md` (this file), `scripts/`, and `projects/BAGEL/README.md` for project goals.
   - Skim `results/log.txt` for the most recent training loop metrics and argument changes.
3. **Artifacts & data sanity**
   - List checkpoints under `results/checkpoints/` (size/date).
   - Inspect dataset config(s), e.g. `projects/BAGEL/data/configs/src_ref_edit.yaml`.
   - If packer is skipping many samples, confirm `expected_num_tokens` vs `max_num_tokens_per_sample`.
4. **Try a smoke test (inference)**
   - Use `projects/BAGEL/scripts/test_ckpt_infer.py` which targets the latest checkpoint by default.
   - If you need to change the path, set `CKPT_PATH` in that script.
5. **If you need to relaunch training**
   - Start with `scripts/run_bagel_train_src_ref_edit.sh` and tune via env vars: `STEPS`, `TOKENS`, `LR`, `NUM_WORKERS`.
   - For clearer errors, consider adding `CUDA_LAUNCH_BLOCKING=1` (and optionally avoid elastic once to expose the raw traceback).
6. **Write a short worklog**
   - Create or append a `notes/` entry with UTC timestamp and a concise summary of what you checked/changed and why.

## Collaboration basics
- Prefer the MCP tools defined in `mcp-fs/server.py` for file access, searches, communication, and job control instead of recreating their functionality.
- Keep commits focused and well-documented so other agents can quickly understand why a change was made.
- When you touch code or configuration, update related documentation within the same change set to keep guidance in sync.

## Knowledge sharing via `notes/`
- Treat `hello-mcp/notes/` as the shared knowledge base for this project. Create new files or append to existing ones rather than rewriting history.
- Start each entry with an ISO-8601 UTC timestamp and a concise summary so future agents can follow the timeline.
- Capture lessons learned, troubleshooting steps, environment tips, and other actionable context that would help successors.
- Maintain an always-current worklog in `hello-mcp/notes/` that documents the active user goal, the plan for satisfying it, and any results or follow-up tasks.
- Add a new note entry whenever the user introduces a new goal or updates an existing goal, and update the worklog as plans evolve or milestones are reached.
- Ensure notes provide enough context that a new contributor can understand the current state of the effort and seamlessly continue the work.
- JSONL logs created by MCP tools (for example `hello-mcp/notes/resource_requests.jsonl` or `hello-mcp/notes/admin_feedback.jsonl`) should only be modified through those tools—do not edit them manually.
- Never store secrets or credentials in `notes/`. If you encounter sensitive information, notify the admins using the feedback tool immediately.

## Communication with admins
- Use the `request_additional_resources` tool to log requests for new permissions, assets, or hardware. Submissions are appended to `hello-mcp/notes/resource_requests.jsonl` for tracking.
- Use the `admin_feedback` tool to share questions, comments, or concerns with the admin team. These entries are logged in `hello-mcp/notes/admin_feedback.jsonl`.
- Keep submissions professional and informative; they are part of the shared knowledge base.

## Safety and consistency
- Automations and scripts must respect the repository layout and should not assume write access outside the sandbox.
- Keep secrets or credentials out of this tree. If any are discovered, stop and report them immediately via `admin_feedback`.
- Avoid leaving ad-hoc notes in code or commit messages—use the shared `notes/` area instead.
