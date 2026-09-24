#!/bin/bash
# after the first paired sweep (pid $1) exits: noise floor -> cross grid
while kill -0 "$1" 2>/dev/null; do sleep 20; done
export PYTHONPATH=/home/lucam/dial-mpc/.claude/worktrees/armature-investigation XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
python3 grid.py --tag noise --plant 1.0 1.4 0.7 --planner match --repeat 2 --seeds $(seq 0 23) --save-rollouts
python3 grid.py --tag cross --plant 1.0 1.4 2.0 0.7 --planner 0.7 0.85 1.0 1.2 1.4 2.0 --seeds $(seq 0 15) --save-rollouts
