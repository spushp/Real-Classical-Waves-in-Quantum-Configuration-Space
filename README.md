# Real Classical Waves in Quantum Configuration Space

Python code reproducing all numerical results and figures (Figs. 1–5, Sec. VIII) of

P. Singh, "Real Classical Waves in Quantum Configuration Space" (submitted to Phys. Rev. A, 2026).

## Requirements
Python 3, numpy, scipy, matplotlib.

## Usage
Each script reproduces one figure (PDF and PNG) and prints the numerical values quoted in the paper:

| Script | Figure | Content |
|---|---|---|
| `fig1_identity.py` | Fig. 1 | Key identity H_t u = −Ω⁻¹∂_t u, energy isometry, conservation of J and E |
| `fig2_schrodinger.py` | Fig. 2 | Schrödinger limit, error bound, mode frequencies, real-wave reconstruction |
| `fig3_config.py` | Fig. 3 | Two-quantum configuration space, multi-time UQM pair, sum/difference frequencies |
| `fig4_nonlinear.py` | Fig. 4 | Cubic Klein–Gordon vs nonlinear Schrödinger, Duffing shift, orientation leakage |
| `fig5_meanfield.py` | Fig. 5 | Coupled real oscillators vs mean field vs exact N-quantum dynamics |

```
python3 fig1_identity.py
```

Units: ħ = c = 1. `common.py` contains the plotting style.

## Citation
If you use this code, please cite the paper above.

## License
GPL-3.0
