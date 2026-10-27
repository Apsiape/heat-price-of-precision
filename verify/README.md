# Finite verification

Run `python verify/run_all.py` from the repository root. Fixed seeds, one BLAS/OpenMP/HiGHS thread, sequential checks. Density matrices have dimension at most 64; a fixed-control linear constraint matrix has size 1024 by 32. No asymptotic theorem is inferred from numerical fits.

The script records:

1. Identical local marginals and quantum/classical/product erasure penalties at 101 probabilities, with explicit CNOT and unitary endpoint checks.
2. CHSH values and the singular-value optimum, including endpoint and intermediate probabilities.
3. Magic-square commutation, parity, all nine question pairs and all 16 outcomes, rank-one local measurements, and the four-unit entropy budget for the thermal working registers of the swap construction. The SWAP channel and its correlations with the answer ports are not simulated. A wrong parity is a negative control.
4. Recovery for small Bell-pair sources by explicit dephasing/partial trace, independent enumeration of communication allocations, and fractional linear programs. Two fixed arithmetic sanity assertions reject assigning two units of savings to one classical bit and assigning zero work to an unbudgeted quantum reunion; they do not simulate a faulty protocol.
5. Finite sector menus solved by linear programming versus two-support enumeration, mutual-information concavity, and dual witnesses.
6. Exact Schur cores, a coherent superposition across two sectors, resource reduction, the W-state support intersection and entropies, and the fixed-control obstruction. A sanity assertion checks that the binary entropy h(1/3) is nonzero.
7. Weighted observer-family optimality, explicit commutator realization with noncommuting probes, finite-reservoir entropy ledger, and multiphase clock/determinant calculations.
8. Change of marginal entropy when one qubit of a pure three-qubit state crosses a cut (the transfer bound for pure states only; no relative entropy of entanglement of a mixed state is evaluated), and the entropy-continuity and hashing bounds of the finite-error conversion on extracted states.

Passing these checks verifies finite identities and examples at the recorded tolerances. It does not prove self-testing, reversible thermodynamic attainability, infinite-dimensional optimization, or the game demand results imported from the cited papers. It does not provide a numerical magic-square robustness constant.

`logs/` contains reproducible outputs and the final build validation.
