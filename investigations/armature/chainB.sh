#!/bin/bash
# after the first paired sweep (pid $1) exits: more seeds -> highest precision -> physical-limit termination
while kill -0 "$1" 2>/dev/null; do sleep 20; done
export PYTHONPATH=/home/lucam/dial-mpc/.claude/worktrees/armature-investigation XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
python3 grid.py --tag paired --plant 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.7 2.0 --planner match 1.0 --seeds $(seq 16 39) --save-rollouts
python3 grid.py --tag paired_hi --precision highest --plant 0.5 0.7 1.0 1.4 2.0 --planner match 1.0 --seeds $(seq 0 15) --save-rollouts
python3 grid.py --tag paired_phys --phys-term --plant 0.5 0.7 1.0 1.4 2.0 --planner match 1.0 --seeds $(seq 0 15) --save-rollouts
