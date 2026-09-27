#!/bin/zsh
# Phase A: refine (8 single-thread runs); phase B: Multiple Oracle (4 runs, 2 threads).
cd "$(dirname "$0")/../.."
# timings are only valid on AC power (macOS slows the CPU on battery)
if [[ "$(uname)" == Darwin ]] && ! pmset -g batt | grep -q "AC Power"; then
  echo "on battery power: plug in first"; exit 1
fi
P=${PYTHON:-python}
for t in 0.03 0.01 0.001 0.0001; do for s in do fp; do
  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 $P -W ignore bench/scripts/refine.py $t $s &
done; done
wait
for n in 2 3 5 10; do NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=1 $P -W ignore bench/scripts/multiplayer.py $n & done
wait
date
