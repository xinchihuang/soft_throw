# Joint Limits Summary (This Repo)

This document consolidates the joint limits currently defined/used in this codebase, split into:

- **URDF soft/hard** limits (position, velocity, effort)
- **Offline planner** software limits (velocity/acceleration/jerk/torque)
- **Online executor** software guards (preflight position check + runtime velocity guard)

Units:

- `q`: rad (arm), m (gripper fingers)
- `qdot`: rad/s (arm), m/s (gripper fingers)
- `qdd`: rad/s² (arm)
- `tau`: Nm (arm), N (gripper fingers effort in URDF)

---

## Sources (where these numbers come from)

In the current repo layout, the **authoritative** software limits live in:

- `core/throw_params.py`
  - `Q_LIMITS_7` (position), `QDOT_LIMITS_7` (velocity), `QDDOT_LIMITS_7` (acceleration)
  - `URDF/PKG` paths used by Pinocchio (when available on the host)

This repo no longer contains the historical folders referenced by older docs (e.g. `planner/`, `executor/`, `ball_throw/`).
If you have an **external** online executor (C++), treat its runtime guards as separate constraints and align
`core/throw_params.QDOT_LIMITS_7`/`Q_LIMITS_7` to those values when generating trajectories for real execution.

---

## Arm joints (j1–j7) — soft vs hard

### Position limits (`q`)

URDF provides both:

- **Soft**: `<safety_controller soft_lower_limit / soft_upper_limit>`
- **Hard**: `<limit lower / upper>`

This repo’s `core/throw_params.py` currently uses **soft** limits (rounded to 3 decimals).

| Joint | `q_soft` lower | `q_soft` upper | `q_hard` lower | `q_hard` upper |
|---|---:|---:|---:|---:|
| j1 | -2.8973 |  2.8973 | -2.9671 |  2.9671 |
| j2 | -1.7628 |  1.7628 | -1.8326 |  1.8326 |
| j3 | -2.8973 |  2.8973 | -2.9671 |  2.9671 |
| j4 | -3.0718 | -0.0698 | -3.1416 |  0.0000 |
| j5 | -2.8973 |  2.8973 | -2.9671 |  2.9671 |
| j6 | -0.0175 |  3.7525 | -0.0873 |  3.8223 |
| j7 | -2.8973 |  2.8973 | -2.9671 |  2.9671 |

### Velocity limits (`qdot`)

URDF hard velocity limits (per-joint `<limit velocity=...>`):

| Joint | `qdot_hard` (URDF) |
|---|---:|
| j1 | 2.175 |
| j2 | 2.175 |
| j3 | 2.175 |
| j4 | 2.175 |
| j5 | 2.610 |
| j6 | 2.610 |
| j7 | 2.610 |

Offline planner uses the same values as software limits:

- `core/throw_params.py`: `QDOT_LIMITS_7 = [2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]`

Online executor runtime guard (measured-velocity E-stop threshold) is **executor-specific** and may be more conservative
than the URDF velocity limits (e.g. some executors use slightly lower thresholds than 2.175/2.610).
If you hit reflex E‑STOPs on real hardware while staying within `QDOT_LIMITS_7`, reduce `QDOT_LIMITS_7` accordingly.

### Acceleration limits (`qdd`) and jerk limits

URDF does **not** define joint acceleration or jerk limits here. This repo defines them as **software limits** for planning:

Planner acceleration limit (`QDDOT_LIMITS_7`, used for time-scaling feasibility in `core/ik_poly5_core.py`):

| Joint | `qdd_soft` (`core/throw_params.QDDOT_LIMITS_7`) |
|---|---:|
| j1 | 15.0 |
| j2 | 7.5 |
| j3 | 10.0 |
| j4 | 12.5 |
| j5 | 15.0 |
| j6 | 20.0 |
| j7 | 20.0 |

This repo does not currently define a single “jerk limit” constant used by the planner.

### Effort / torque limits (`tau`)

URDF hard effort limit (per-joint `<limit effort=...>`):

| Joint | `tau_hard` (URDF effort) |
|---|---:|
| j1 | 87 |
| j2 | 87 |
| j3 | 87 |
| j4 | 87 |
| j5 | 12 |
| j6 | 12 |
| j7 | 12 |

This repo does not currently implement a torque-limited planner, and the online executor is out-of-tree.
Treat URDF effort values as reference only unless your executor enforces them.

---

## Gripper finger joints (optional DOF in URDF)

URDF defines hard limits for the prismatic fingers; no soft safety_controller tags are present.

| Joint | type | `q_hard` lower | `q_hard` upper | `qdot_hard` | `effort_hard` |
|---|---|---:|---:|---:|---:|
| panda_finger_joint1 | prismatic | 0.0000 | 0.0400 | 0.2 | 20 |
| panda_finger_joint2 | prismatic | 0.0000 | 0.0400 | 0.2 | 20 |

---

## Notes / interpretation

- “Soft” vs “hard” in this repo mainly applies to **position** (`q`) due to the URDF containing both safety soft limits and hard joint limits.
- Velocity/effort “soft” values in the tables above refer to **software-layer limits** (planner/executor guards) which are set equal to the URDF hard limits in the current code.
- Acceleration/jerk limits are **planner-only** constants; they are not enforced by the executor at runtime.
