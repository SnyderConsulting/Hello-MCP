# Agent Onboarding & Operating Contract (for stateless LLM agents using Runpod MCP)

> **Why this exists**
> I don’t persist between replies. I can’t run timers or do work after I respond unless I use tools that *you* provide. This workspace compensates with Runpod MCP so I can kick off long GPU jobs, log their IDs, and hand them off cleanly.

---

## 0) The Model Contract — what I can / can’t do

**I can:**
- Start long-running processes with `job_start` and return `{job_id, log_path, log_url}` immediately.
- Tail logs in the current reply using `job_logs(..., tail_lines=N, follow_ms=25000)` for a short window.
- Read/write files inside FS_ROOT with `read_file`, `write_file`, `list_dir`, `grep`.
- Inspect GPUs with `gpu_info`.

**I cannot (without an explicit trigger from you):**
- Keep working after I send a message.
- Promise to monitor/finish a job later.
- Poll on my own schedule. (I can do it *if you ask each time*, or you create a scheduled task outside of me.)

> **Call-out:** If ongoing monitoring is desired, create a scheduled checker (outside the model) and have it ping me to fetch `job_status` / `job_logs` when needed.

---

## 1) Tools this workspace gives me (the compensations)
- `job_start` — launch heavy work detached; returns an ID and log locations. Optional timeouts.
- `job_status` / `job_list` — check state later by ID.
- `job_logs` — stream/tail logs; `follow_ms` allows ~25s of waiting in one reply.
- `job_stop` — send signals (TERM/KILL) to runaway jobs.
- `list_dir`, `read_file`, `write_file`, `grep` — file & content ops in the jail.
- `gpu_info` — prove an A100 80GB is present and free.
- `notes/` — shared, append-only project context and handoffs. (I should write brief worklogs here.)

---

## 2) First 5 minutes for any new agent
1. **State the contract** in your first message ("I’m stateless; I’ll use MCP to launch and hand off").
2. **Environment sanity:** `gpu_info`, confirm CUDA/BF16 availability, and show device inventory.
3. **Locate artifacts:** enumerate `results/checkpoints/`, `projects/BAGEL/models/BAGEL-7B-MoT/`.
4. **Pick the runner** (e.g., `scripts/edit_infer.py` for source→reference edits) and validate inputs exist.
5. **Plan and kickoff:** outline args, then `job_start` with a clear name; capture `job_id`.
6. **Handoff note:** append a short entry in `notes/` with goal, job_id, paths, and next steps.

---

## 3) Heavy-job playbook
**Kickoff (within one reply):**
- Explain what will run and where outputs go.
- Start with `job_start` and *echo* the `job_id` and `log_url`.
- Tail the last ~200 lines once to verify progress; paste a small excerpt.
- Write a worklog entry in `notes/` (UTC timestamp) with: 
  - purpose, command/args, `job_id`, `log_path`, output dir, stop conditions.

**Status checks (follow-ups):**
- On request, call `job_status` and `job_logs(..., tail_lines)`; summarize.
- If a job is stuck or noisy, suggest `job_stop` with context.

**Completion:**
- Verify expected artifacts exist (e.g., `outputs/edited.png`, metrics files).
- Put a concise summary + pointers in `notes/`, mark blockers if not done.

---

## 4) Response templates agents can copy

**A. "Starting a long job"**
> I can’t keep running after this reply, so I’m launching your job via MCP and handing off the ID so anyone (or I in a later message) can check it.
> **Plan:** `<one line>`
> **Outputs:** `<paths>`
> **Command:** `<args or script>`
> **Launched:** `job_id=...`, `log_url=...`
> **Next check:** Ask me to "check job ..." anytime, or have a scheduler ping me to fetch logs.

**B. "Status check"**
> `job_id=...` — state: `<running|finished|failed>`; last N log lines show `<summary>`. Artifacts at `<paths>`. Want me to stop it or tail more?

**C. "Handoff note" (to append in `notes/`)**
```
2025-09-20T17:04:00Z — BAGEL edit inference kickoff
job_id=..., cmd=python scripts/edit_infer.py ...
artifacts=projects/BAGEL/outputs/edited.png
next: if success, evaluate sharpness; if fail, lower cfg_image_scale to 1.2 and retry.
```

---

## 5) Do / Don’t (to reduce confusion)
**Do**
- Always repeat the contract when starting/monitoring long jobs.
- Always surface `job_id` + `log_url` + output paths.
- Write/append a short `notes/` entry for every kickoff or status milestone.

**Don’t**
- Promise to "keep monitoring" without a concrete trigger.
- Give time estimates; just state how to check and what "done" looks like.

---

## 6) Example: BAGEL reference→source edit (step-50 checkpoint)
**Goal.** Apply a reference image to a source image using our step-50 finetune while preserving identity/pose/background; save to `projects/BAGEL/outputs/edited.png`.

**Runner.** `projects/BAGEL/scripts/edit_infer.py` (loads the full BAGEL, overlays our checkpoint with `strict=False`).

**Definition of done.**
- Script runs on GPU without exceptions.
- `outputs/edited.png` exists.
- Edit blended naturally; CFG tweaks produce sensible changes.

---

## 7) Escalation & admin comms
- Use `request_additional_resources` for quota/permission needs.
- Use `admin_feedback` to flag environment issues or risky data.

---

*This file is a concise companion to `AGENTS.md`. Place a short banner (below) at the top of `AGENTS.md` linking here so new agents see the model contract first.*

> **Agent Model Contract (TL;DR)** — I’m stateless. I can launch long GPU jobs via MCP, return a `job_id`, tail a bit of logs now, and hand off. I won’t promise to keep monitoring unless you (a) ask again later or (b) set up an external scheduler to ping me. See `AGENT_ONBOARDING.md` for the full playbook.
