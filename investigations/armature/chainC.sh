#!/bin/bash
# after pid $1 exits: paired nominal-vs-matched with implicit joint damping (causal test of the
# discretisation explanation for the low-armature gap)
while kill -0 "$1" 2>/dev/null; do sleep 20; done
export PYTHONPATH=/home/lucam/dial-mpc/.claude/worktrees/armature-investigation XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
python3 grid.py --tag paired_impl --implicit-damping --plant 0.5 0.6 0.7 1.0 1.4 2.0 --planner match 1.0 --seeds $(seq 0 15) --save-rollouts
