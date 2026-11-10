Supplemental code for "Real Classical Waves in Quantum Configuration Space"
P. Singh, submitted to Proceedings of the Royal Society A.

Requirements: Python 3, numpy, scipy, matplotlib
(tested with numpy 2.5, scipy 1.18, matplotlib 3.11).
Each script reproduces one figure (PDF and PNG) and prints the numerical values quoted in section 8:

  fig1_identity.py     Figure 1  key identity H_t u = -Omega^{-1} d_t u, energy isometry, J and E conservation   (~20 s)
  fig2_schrodinger.py  Figure 2  Schrodinger limit, error bound, mode frequencies, real-wave reconstruction     (~2 s)
  fig3_config.py       Figure 3  two-quantum configuration space, multi-time UQM pair, sum/difference frequencies (~2 s)
  fig4_nonlinear.py    Figure 4  cubic Klein-Gordon vs nonlinear Schrodinger, Duffing shift, orientation leakage (~40 s)
  fig5_meanfield.py    Figure 5  coupled real oscillators vs mean field vs exact N-quantum dynamics            (~10 s)

common.py holds the plotting style. Units: hbar = c = 1. Notation follows the paper
(eta = Kerr coefficient, lambda = mean-field coupling, a = envelope/oscillation amplitude).
Run, e.g.:  python3 fig1_identity.py
