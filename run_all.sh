#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/experiments"
mkdir -p ../results
run() { echo "== $*"; python3 "$@" | tee "../results/log_${1%.py}.txt"; }
run exp_hopf.py
run exp_waves.py 2.0 48 4
run exp_multistability.py
run exp_phasediff.py
run exp_protocol.py
run exp_table144.py
run exp_route.py 5
run exp_route_chaotic.py
run exp_route_refine.py
run exp_pd_route.py
run exp_chaos.py
run exp_upo_full.py
run exp_saddle.py
run exp_grid.py
run exp_surface.py
run exp_lyap_accuracy.py
rm -f ../results/bench_wave.json
run exp_benchmark.py all
run make_figures.py
run make_tables.py hopf t144 route pd bench
