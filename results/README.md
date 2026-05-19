# Pipeline Evaluation Results

Cleaned per-trial CSV exports for the thesis appendix (Appendix B).  
Each file is derived from [`pipeline_eval_ros2`](../pipeline_eval_ros2/README.md) logger output and matches the trial rows in the appendix tables. Scenario-level aggregates (**N**, **PSR**, **ESR**, mean CER, mean Rec./LLM/CPU/Mem) are computed from these trials when building the LaTeX tables; they are not stored as separate rows in the CSV.

## Files

| File | Source | Evaluation | Rows |
|------|--------|------------|------|
| [`pipeline_eval_video.csv`](pipeline_eval_video.csv) | `pipeline_eval_metrics0.csv` (`asl_mode=video`) | B.1 Video-based | 45 |
| [`pipeline_eval_text.csv`](pipeline_eval_text.csv) | `pipeline_eval_metrics0.csv` (`asl_mode=text`) | B.2 Text-injection | 71 |
| [`pipeline_eval_camera_appendix.csv`](pipeline_eval_camera_appendix.csv) | `pipeline_eval_metrics_from_robot.csv` (`asl_mode=camera`) | B.3 Robot-camera (published trials) | 30 |

Rows are sorted in the same order as the appendix tables: scenario block first (summary metrics in the thesis), then trials within each scenario.

## Columns

### Per-trial fields (in this directory)

| Column | Type | Description |
|--------|------|-------------|
| `evaluation` | string | `video`, `text`, or `camera` (filter label) |
| `gt_text` | string | Ground-truth scenario label (appendix: Scenario) |
| `recognized_text` | string | ASR output (video/camera) or injected command (text) |
| `cer` | float | Character error rate; empty or unused for text-injection |
| `plan_success` | int | Per-trial plan outcome (`1` / `0`; appendix: P) |
| `execution_success` | int | Per-trial execution outcome (`1` / `0`; appendix: E) |
| `asl_total` | float | Recognition time in seconds (appendix: Rec.) |
| `llm_total` | float | Planner latency in seconds (appendix: LLM) |
| `cpu_percent_done` | float | CPU use at trial completion (%) |
| `mem_percent_done` | float | Memory use at trial completion (%) |
| `stop_reason` | string | Logger stop reason; empty when not reported or on normal completion |

**Stop reason normalization**

- Cleared when `plan_success=1`, `execution_success=1`, and the logger reported `goal_reached`.
- `plan_skipped:no_valid_plan` → `No valid plan`; clarification skips → `Parameters not specified`.