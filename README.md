# Visual Pick-and-Place with DAgger

Visuomotor imitation learning for a Franka Panda sorting task in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). A student policy learns from RGB cameras and proprioception to pick colored cubes and place them in matching bins. Training uses **DAgger** (Dataset Aggregation) to correct **covariate shift**—the compounding error that occurs when a policy trained only on expert trajectories visits states the expert never demonstrated.

This repository implements a full DAgger loop inside Isaac Sim: beta-mixed rollouts, oracle relabeling, growing aggregate datasets, and incremental policy updates via [Robomimic](https://robomimic.github.io/).

---

## The Problem DAgger Solves

In visuomotor manipulation, a policy trained purely on expert demonstrations suffers from **distribution shift**. The student executes its own (imperfect) actions, drifts into unfamiliar states, and small errors compound. DAgger addresses this iteratively:

1. **Roll out** the current student policy in the environment.
2. **Query** an expert oracle for the correct action at every visited state.
3. **Aggregate** those `(observation, expert_action)` pairs into a growing dataset.
4. **Retrain** the policy on the full aggregate dataset.

Over iterations, the dataset covers states the student actually encounters, while labels remain expert-quality. Expert intervention during rollouts (controlled by **β**) keeps the robot in recoverable configurations early in training.

---

## Task

A tabletop scene contains:

- **Franka Panda** arm with a parallel-jaw gripper
- **Two colored cubes** (blue, red) randomized on the table
- **Two sorting bins** (blue, red) randomized in disjoint workspace regions
- **Two RGB cameras**: a fixed table camera and a wrist-mounted camera (256×256)

**Goal:** place each cube in its color-matched bin. Success is detected when both cubes are within spatial tolerances of their target bins.

**Challenge:** the student sees only cameras and proprioception—no object poses. The expert oracle uses privileged ground-truth state available only during data collection, not at deployment.

---

## System Architecture

```mermaid
flowchart TB
    subgraph seed["Initialization"]
        ED[Expert demonstration HDF5]
        AGG[(Aggregate dataset)]
        CKPT[Warm-start policy checkpoint]
        ED --> AGG
    end

    subgraph iter["DAgger iteration i"]
        ROLL[Rollout collector]
        STU[Student policy π_i]
        EXP[IK expert oracle]
        MIX["β-mixed action<br/>expert or student"]
        RELABEL["Label every step<br/>with expert action"]
        APPEND[Append to aggregate]
        TRAIN["Fixed gradient steps<br/>on full aggregate"]
        CKPT_NEW[Checkpoint π_{i+1}]

        STU --> ROLL
        EXP --> ROLL
        ROLL --> MIX
        MIX --> RELABEL
        RELABEL --> APPEND
        APPEND --> AGG
        AGG --> TRAIN
        CKPT --> TRAIN
        TRAIN --> CKPT_NEW
    end

    seed --> iter
    CKPT_NEW --> iter
```

### Path A DAgger (this implementation)

This project follows **Path A** DAgger as described in the original algorithm:

| Design choice | Rationale |
|---|---|
| **Beta-mixed execution** | With probability β, execute the expert action; otherwise execute the student. Keeps rollouts on-manifold early while still exposing the student to its own mistakes. |
| **Expert labels on every step** | Regardless of who executed the action, store the expert's action as the training label. This is the core DAgger insight: learn what the expert *would* do in states the student visits. |
| **Fixed gradient steps per iteration** | Each round runs a fixed number of supervised updates on the entire aggregate dataset, rather than training to convergence. Prevents overfitting to the latest batch and keeps iteration cost predictable. |
| **Growing aggregate HDF5** | All data from every iteration is retained. The dataset monotonically grows, preserving coverage of earlier state distributions. |

---

## DAgger Loop in Detail

### 1. Seed the aggregate dataset

The aggregate dataset starts as a copy of an expert demonstration HDF5 file (`ik_expert_demos_vis_robomimic.hdf5`). These trajectories were collected with the same observation layout the student uses at deployment (proprioception + `table_cam` + `wrist_cam`), ensuring the initial policy has a reasonable starting distribution.

### 2. Roll out with β-mixing

For each DAgger iteration `i`, a mixing coefficient βᵢ is computed from the schedule. At every timestep:

```
rollout_action = expert_action   if random() < βᵢ
                 student_action  otherwise
```

The environment steps with `rollout_action`, but the stored training label is always `expert_action` queried from privileged state.

### 3. Append labeled trajectories

Collected episodes are written to the aggregate HDF5 in Robomimic format. Each transition stores student observations paired with expert actions.

### 4. Policy update

The visuomotor policy is updated for a **fixed number of gradient steps** on the full aggregate dataset. The model weights carry over between iterations (warm-started from the previous checkpoint). A new checkpoint `dagger_iter_<N>.pth` is saved after each round.

### 5. Repeat

The updated checkpoint becomes the rollout policy for the next iteration. β decreases over time, shifting execution from expert-heavy to student-heavy.

---

## Expert Oracle

The registered expert solver `ik_pick_place` is a **privileged, model-based controller**—not a learned policy. It is used only during data collection to provide ground-truth action labels.

### Pipeline

1. **Parse privileged state** — object and bin poses from simulator ground truth, mapped to color-matched pick-place pairs.
2. **Plan waypoints** — a finite-state machine (`APPROACH_PICK → DESCEND_PICK → GRASP → LIFT → APPROACH_PLACE → DESCEND_PLACE → RELEASE → LIFT_AFTER`) generates ordered TCP targets with independent pick/place height offsets.
3. **Track with differential IK** — a DLS-based differential IK controller converts pose targets to joint commands each step.
4. **Gripper logic** — binary open/close based on FSM phase, with dwell timers at grasp and release for reliable contact.

The expert never sees camera images. Privileged observations are enabled only in the DAgger environment configuration; the student policy group remains visuomotor-only.

### Why a scripted IK expert?

- **Deterministic, near-perfect labels** — DAgger's theoretical guarantees assume access to a correct oracle. A scripted controller with ground-truth poses provides consistent supervision without label noise from a second learned model.
- **Decouples oracle quality from student quality** — the expert does not degrade as the student improves.
- **Reproducible** — same scene layout always produces the same expert behavior, making ablations and debugging tractable.

---

## Student Policy (Learner)

The student is a **BC-RNN visuomotor policy** trained through Robomimic's supervised learning interface. Within DAgger, it serves as the rollout policy and is incrementally updated each iteration. The architecture is fixed across all DAgger rounds:

| Component | Setting | Justification |
|---|---|---|
| **Visual encoder** | ResNet18Conv + SpatialSoftmax (16 keypoints) | Standard visuomotor backbone from Robomimic; spatial softmax yields a compact, spatially grounded representation without heavy compute. |
| **RGB inputs** | `table_cam`, `wrist_cam` | Table camera provides global context for bin/object layout; wrist camera provides close-range grasp alignment. Both are 256×256 uint8. |
| **Proprioception** | 18D (joint pos + vel, relative) | Gives the policy short-horizon dynamics information cameras alone cannot provide. |
| **Recurrence** | LSTM, hidden dim 400, horizon 10 | Manipulation is temporally extended; a 10-step context captures gripper state and motion history without the cost of a Transformer. |
| **Action space** | 8D relative joint deltas + binary gripper | Matches expert demonstration format; relative actions are easier to learn and generalize across joint configurations. |
| **Loss** | L2 on actions | Deterministic regression is appropriate for a near-deterministic expert. |
| **Optimizer** | Adam, lr = 1×10⁻⁴ | Conservative learning rate for stable updates on a growing dataset across iterations. |
| **Batch size** | 32 | Fits GPU memory with dual RGB streams and LSTM; large enough for stable gradients. |
| **Sequence length** | 10 | Aligned with RNN horizon for consistent temporal batches. |

---

## Parameter Reference and Design Rationale

### DAgger training parameters

| Parameter | Default | Justification |
|---|---|---|
| `--num_iterations` | `5` | Five rounds balance dataset growth against compute. Each iteration adds on-policy states; diminishing returns typically appear after a handful of rounds for tabletop tasks with moderate horizon. |
| `--episodes_per_iteration` | `20` | Enough rollouts per round to sample diverse randomized cube/bin poses (see environment randomization ranges) without making a single iteration prohibitively slow in simulation. |
| `--grad_steps_per_iteration` | `5000` | Fixed update budget per iteration (Path A DAgger). Chosen to be substantial enough to incorporate new data while avoiding overfitting to the latest batch. With batch size 32, this is ~160K samples seen per iteration. |
| `--beta_schedule` | `inv_sqrt` | βᵢ = 1/√i. The inverse-square-root schedule is a standard DAgger choice: β=1.0 on iteration 1 (pure expert rollouts, safe exploration), β≈0.71 on iteration 2, β≈0.58 on iteration 3, etc. It decreases expert intervention smoothly without collapsing to zero too quickly. |
| `--beta_schedule linear` | βᵢ = 1 − (i−1)/N | Alternative that reaches β=0 by the final iteration. Useful when you want the last round to be fully on-policy with no expert assistance. |
| `--horizon` | `2500` | Maximum rollout steps. The IL environment allows 45 s episodes at 60 Hz effective control (120 Hz sim, decimation 2), yielding ~2700 steps. 2500 provides headroom while capping runaway rollouts from a failing student. |
| `--seed_dataset` | `ik_expert_demos_vis_robomimic.hdf5` | Visuomotor expert demos with matching observation keys. Seeds the aggregate with off-policy expert coverage before any student rollouts. |
| `--aggregate_dataset` | `dagger_aggregate.hdf5` | Monotonically growing dataset path. Persists across iterations and resume. |
| `--solver` | `ik_pick_place` | Registered IK expert for color-matched sorting. |
| `--ee_speed` | `0.75` m/s | End-effector waypoint tracking speed. Fast enough for reasonable episode length (~30–60 s per successful sort) but slow enough for stable grasping and placement with differential IK. |
| `--batch_size` | `32` (from config) | Override via CLI if GPU memory requires it. |
| `--enable_cameras` | required | DAgger rollouts must render RGB observations for the visuomotor student. |

### Expert solver parameters (internal defaults)

| Parameter | Default | Justification |
|---|---|---|
| `safe_clearance` | `0.15` m | Hover height above tallest scene object/bin during transit. Prevents collisions during horizontal moves. |
| `pick_clearance` | `−0.03` m | Slight downward offset at pick (negative = below cube center Z). Compensates for gripper finger geometry so the TCP reaches grasp depth. |
| `place_clearance` | `0.05` m | Release height above bin floor. Ensures the cube drops into the bin rather than dragging along the rim. |
| `waypoint_tol` | `0.05` m | General waypoint reach threshold for lift/transit phases. |
| `pick_waypoint_tol` | `0.025` m | Tighter tolerance for pick approach/descent. Precise XY alignment is critical for reliable grasps. |
| `place_xy_tol` | `0.05` m | Horizontal tolerance for release readiness over the bin. |
| `place_z_tol` | `0.06` m | Vertical tolerance for release readiness. |
| `grasp_dwell_steps` | `15` | Timesteps to hold closed at pick before lifting. Allows contact forces to settle. |
| `release_dwell_steps` | `25` | Timesteps to hold open at place before retreating. Ensures the cube is released before the arm moves away. |

### Environment parameters

| Parameter | Value | Justification |
|---|---|---|
| `episode_length_s` | `45.0` | Two sequential pick-place cycles with IK expert take ~30–45 s. Generous timeout prevents premature truncation during student rollouts. |
| `decimation` | `2` | Control at 60 Hz (120 Hz physics). Standard for manipulation; balances sim fidelity and throughput. |
| Arm action | Relative joint position, scale 1.0 | Matches expert demo format. Relative deltas are invariant to absolute configuration and easier for both IK and learned policies. |
| Gripper action | Binary open/close | Simplifies the action space; sufficient for parallel-jaw grasping. |
| Robot PD gains | stiffness 400, damping 80 | High-gain tracking for reliable IK execution and crisp learned policy behavior. |
| `disable_gravity` (robot) | `True` | Reduces unintended base drift during precise manipulation in IL settings. |
| Camera resolution | 256×256 | Practical balance between visual detail and GPU memory during DAgger rollouts with dual cameras. |
| Table camera FOV | wide (focal length 10 mm) | Keeps the full workspace in frame for global spatial reasoning. |
| Cube randomization | X: 0.58–0.72, Y: ±0.14, min separation 0.14 m | Objects spawn in the manipulable region with collision-free placement. |
| Bin randomization | Disjoint Y bands (blue: −0.32 to −0.18, red: 0.18 to 0.32) | Bins cannot overlap; forces generalization across bin positions. |
| Success tolerance | XY: 0.07 m, Z: 0.05 m | Cubes must be clearly inside bins, not merely nearby. |

### Robomimic training config (used within each DAgger iteration)

| Parameter | Value | Justification |
|---|---|---|
| `hdf5_cache_mode` | `low_dim` | Cache proprioception in memory; stream RGB from disk. Reduces RAM pressure as the aggregate dataset grows with image data. |
| `hdf5_normalize_obs` | `false` | Images are uint8; proprioception is already well-scaled. Normalization adds complexity without benefit here. |
| `num_data_workers` | `4` | Parallel data loading to keep the GPU fed during 5000-step update phases. |
| `crop_randomizer` | 224×224 crop from 256×256 | Mild data augmentation on RGB; standard in Robomimic visuomotor pipelines. |
| `pretrained` (ResNet18) | `false` | Train from scratch on domain-specific sim images; ImageNet features are a poor match for Isaac Sim renders. |

---

## Prerequisites

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (conda or uv install recommended)
- NVIDIA GPU with Isaac Sim support
- Python environment with Isaac Lab dependencies

Install this extension in editable mode:

```bash
python -m pip install -e source/visual_pick_and_place
```

Verify registration:

```bash
python scripts/list_envs.py
```

You should see `Template-Visual-Pick-And-Place-IL-Dagger-v0` among the registered tasks.

---

## Running DAgger

### Step 1 — Collect expert seed demonstrations

Collect visuomotor expert trajectories with the IK solver. Only successful episodes (both cubes in matching bins) are saved.

```bash
python scripts/collect_expert_demos.py \
  --task Template-Visual-Pick-And-Place-v0 \
  --solver ik_pick_place \
  --num_episodes 50 \
  --record_cameras \
  --dataset_file ./datasets/ik_expert_demos.hdf5 \
  --ee_speed 0.75 \
  --enable_cameras
```

Prepare the HDF5 for Robomimic (restructures observations under `obs/`):

```bash
python scripts/prepare_robomimic_dataset.py \
  --input ./datasets/ik_expert_demos.hdf5 \
  --output ./datasets/ik_expert_demos_vis_robomimic.hdf5
```

### Step 2 — Obtain a warm-start checkpoint

DAgger requires an initial policy checkpoint to begin rollouts. This should be a policy trained on the seed dataset (at minimum enough training to produce coherent, if imperfect, motions). Provide it via `--checkpoint`.

### Step 3 — Run DAgger training

```bash
python scripts/train_dagger.py \
  --task Template-Visual-Pick-And-Place-IL-Dagger-v0 \
  --checkpoint logs/.../models/warmstart.pth \
  --seed_dataset ./datasets/ik_expert_demos_vis_robomimic.hdf5 \
  --aggregate_dataset ./datasets/dagger_aggregate.hdf5 \
  --solver ik_pick_place \
  --num_iterations 5 \
  --episodes_per_iteration 20 \
  --grad_steps_per_iteration 5000 \
  --beta_schedule inv_sqrt \
  --horizon 2500 \
  --ee_speed 0.75 \
  --enable_cameras
```

**Resume** from a previous run (continues from the next iteration):

```bash
python scripts/train_dagger.py \
  --task Template-Visual-Pick-And-Place-IL-Dagger-v0 \
  --resume logs/dagger/.../models/dagger_iter_3.pth \
  --num_iterations 5 \
  --episodes_per_iteration 20 \
  --grad_steps_per_iteration 5000 \
  --horizon 2500 \
  --enable_cameras
```

### Step 4 — Evaluate

```bash
python scripts/play_robomimic_bc.py \
  --task Template-Visual-Pick-And-Place-IL-Visuomotor-v0 \
  --checkpoint logs/dagger/.../models/dagger_iter_5.pth \
  --num_rollouts 10 \
  --horizon 2500 \
  --enable_cameras
```

Use the IL visuomotor task (no privileged observations) for evaluation that matches deployment conditions.

---

## Outputs and Logging

Each DAgger run creates a timestamped directory under `logs/dagger/`:

```
logs/dagger/<task>/<timestamp>/
├── config.json                  # Robomimic config snapshot
├── logs/
│   ├── dagger_run_state.json    # Resume metadata (iteration, paths, β schedule)
│   ├── dagger_summary.json      # Per-iteration metrics (β, success rate, dataset size)
│   ├── dagger_iter_<N>_metrics.json
│   └── dagger_iter_<N>_rollouts.hdf5   # optional debug export
└── models/
    └── dagger_iter_<N>.pth      # Policy checkpoint after iteration N
```

Key metrics logged per iteration:

- **β** — expert mixing coefficient
- **Success rate** — fraction of student rollouts that completed the task
- **Aggregate demos** — total episodes in the growing dataset
- **Training loss** — supervised loss over the fixed gradient steps

---

## Project Structure

```
visual_pick_and_place/
├── scripts/
│   ├── collect_expert_demos.py      # Expert demonstration collection
│   ├── prepare_robomimic_dataset.py # HDF5 restructuring for Robomimic
│   ├── train_dagger.py              # DAgger training entry point
│   └── play_robomimic_bc.py         # Policy evaluation
└── source/visual_pick_and_place/visual_pick_and_place/
    ├── experts/
    │   ├── dagger_collector.py      # β-mixed rollouts with expert labeling
    │   ├── solvers/ik_pick_place_solver.py
    │   └── ik/                      # Trajectory planning, kinematics, task parsing
    ├── robomimic_training/
    │   ├── dagger_trainer.py        # Fixed-step aggregate training
    │   ├── dagger_dataset.py        # HDF5 aggregate management
    │   ├── dagger_resume.py         # Checkpoint resume logic
    │   └── policy_inference.py      # Student rollout inference
    └── tasks/.../
        ├── visual_pick_and_place_dagger_env_cfg.py  # Privileged obs for expert
        └── visual_pick_and_place_il_env_cfg.py      # Student visuomotor obs
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation | Isaac Lab / Isaac Sim |
| Robot | Franka Panda (7-DOF + parallel gripper) |
| IL framework | Robomimic (BC-RNN) |
| Algorithm | DAgger (Dataset Aggregation) |
| Expert | Differential IK + FSM waypoint planner |
| Data format | HDF5 (Robomimic-compatible) |
| Policy | ResNet18 + SpatialSoftmax + LSTM |

---

## Key Design Decisions (Summary)

1. **Privileged expert, visuomotor student** — the oracle uses ground-truth poses; the deployable policy uses only cameras and proprioception. This mirrors real-world IL where experts have more information during collection than the deployed agent.

2. **Path A with fixed gradient steps** — predictable per-iteration cost and stable learning dynamics on a monotonically growing dataset.

3. **Inverse-square-root β schedule** — gradual handoff from expert-driven to student-driven rollouts without abrupt distribution shifts between iterations.

4. **Relative joint actions** — consistent action space across expert and student, simplifying both IK control and learned policy regression.

5. **Dual-camera visuomotor input** — global table view for spatial layout plus wrist view for precision manipulation.

6. **Success-filtered expert seeds** — the initial dataset contains only successful expert trajectories, giving DAgger a clean starting distribution before on-policy correction begins.

---

## References

- Ross, Gordon, & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (2011) — original DAgger algorithm
- [Robomimic](https://robomimic.github.io/) — imitation learning framework
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — GPU-accelerated robot learning in simulation
