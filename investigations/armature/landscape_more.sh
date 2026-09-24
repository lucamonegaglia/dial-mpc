#!/bin/bash
# extra landscape seeds with capped GPU memory (the first batch beyond seed 1 died of cuSolver OOM)
export PYTHONPATH=/home/lucam/dial-mpc/.claude/worktrees/armature-investigation XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
for s in "$@"; do
  python3 landscape.py --plants 0.5 0.7 1.0 1.4 1.7 2.0 --models 0.7 1.0 1.4 2.0 --n-states 20 --every 4 --seed "$s" --out "landscape_s$s"
done
