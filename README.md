# Sublattice Bloch Reduction for Lattice Dynamical Systems

This repository contains our implementation of experiments and scripts for our research
*Sublattice Inheritance and Multistability in Chaotic Lattice Dynamics*. 

## Running the experiments

The requirements are Python 3.12 (tested with 3.12.3), NumPy 2.3.1, SciPy 1.16.0, Matplotlib 3.10.8, and `epstopdf`. All scripts set `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to one, so all timings are single threaded. 
```bash
./run_all.sh
```
`run_all.sh` runs the experiments in dependency order, writes JSON files and logs to `results/`,
the figures (EPS) to `figures/`, and prints the rows of the tables. Each script can also
be run on its own from inside `experiments/`, for example `python3 exp_hopf.py`. The script
`exp_benchmark.py` accepts `wave`, `lyap`, `orbit` or `all`; it resumes an existing
`results/bench_wave.json`, which `run_all.sh` deletes first so that all timings are measured again.

## Package `sublattice/`

| Module | Content |
| --- | --- |
| `models.py` | Lorenz-96 lattice and diffusive rings of Lorenz, Rössler and Hindmarsh–Rose oscillators: vector field, local Jacobian blocks $a_s(j;x)$, fused evaluation of both |
| `bloch.py` | stencil phases, twisted operators $A_\theta$, Runge–Kutta propagation of the state with all twisted variational blocks, reduced Bloch monodromy $R_\theta=P_\theta^{-1}\Phi_\theta(\tau)$ and multipliers $e^{i\theta\kappa}\nu^d$ (Algorithm 2), Bloch symbols, Bloch–Lyapunov spectra (Algorithm 3) |
| `waves.py` | settling, minimal spatial period, detection of rotating waves, reduced shooting for primitive waves (Algorithm 1), standard periodic shooting, period detection |
| `hopf.py` | first Hopf bifurcation of the homogeneous state, superlevel sets of $g(\theta)=\cos\theta-\cos 2\theta$, inheritance horizons of Proposition 12 |
| `admissible.py` | Farey phases, unstable denominators, admissible multiples as an order ideal, simplest fractions by the Stern–Brocot tree |
| `phasediff.py` | drift $c_P$ and phase diffusion coefficient $D_P$ in the Dyson form with the group inverse and in the Green function form |
| `baselines.py` | the nine compared methods: SBR, FM (full monodromy), RM (reduced monodromy in the full lattice), IRR (projection operators of irreducible representations), SBD (simultaneous block diagonalization of Zhang and Motter), NK (ARPACK on the monodromy operator), HILL (Floquet–Fourier–Hill), BQR (discrete QR), CQR (continuous QR) |

`common.py` holds the shared helpers: thread settings, paths, JSON input and output, the timing and
memory measurement (time as the minimum over repeated runs, peak memory traced with `tracemalloc` in a
separate run) and the search for stable rotating waves.

## Minimal example

```python
import numpy as np
from sublattice.models import Lorenz96
from sublattice.waves import settle, detect_rotating_wave
from sublattice.baselines import sbr_wave, fm_wave
from sublattice.bloch import wave_bloch_multipliers

model = Lorenz96(2.0)
rng = np.random.default_rng(1009)
x = settle(model, 2.0 + 0.5 * rng.standard_normal((9, 1)), t=600.0)
kappa, tau, z = detect_rotating_wave(model, x)
mu_sbr = sbr_wave(model, z, tau, kappa, 64, 100)
mu_fm = fm_wave(model, z, 9 * tau, 64, 900)
symbol = np.max(np.abs(wave_bloch_multipliers(model, z, tau, kappa, 100, np.linspace(0.1, np.pi, 50))), axis=1)
```

`sbr_wave` returns all $9\cdot 64=576$ multipliers of the lifted wave from integrations on nine sites over
one ninth of the period, and `fm_wave` returns the same multipliers from the full monodromy matrix.
