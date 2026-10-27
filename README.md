# The heat price of precision

Seth Douglas and Nidhal Mghirbi. Version 1.0.0, October 2026. DOI: [10.5281/zenodo.23701732](https://doi.org/10.5281/zenodo.23701732).

The paper asks when the entanglement needed for precise nonlocal-game play becomes heat. For fixed games whose usable-entanglement demand diverges, local reset before communication forces divergent work, while quantum reunion before reset permits bounded operating work. The manuscript proves the sharp leading law for the imported game family, a quantum/classical block-recovery tradeoff, an exact-matching sector theorem, and a four-unit magic-square cycle.

- [PDF](paper/main.pdf)
- [LaTeX source](paper/main.tex)
- [Verification guide](verify/README.md)
- [Recorded numerical output](verify/logs/checks.txt)

All results use the thermodynamic rate and reset contract in Section 1. The arbitrary-schedule identity is operational and does not solve every causal optimization. The sector theorem requires exact matching. Laboratory implementations and the unrestricted stability extension are explicitly labelled SKETCH and CONJECTURE.

## Build and verify

Requires Python 3.10+, NumPy, SciPy, psutil, threadpoolctl, and a TeX distribution with pdflatex, BibTeX, and plainurl.bst.

```text
python verify/run_all.py
python paper/build.py
```

Scripts set numerical thread counts to one before importing libraries. Checks use small matrices and linear programs, report peak memory, and reject 4 GiB usage. The build retains final TeX/BibTeX logs and rejects warnings, undefined references, and overfull or underfull boxes in the final pass. Intermediate build and page-rendering files are ignored by Git; the PDF and verification logs are committed.

## How this work was done

The authors developed this work with current frontier AI models from more than one developer and take responsibility for every claim. The section "How this work was done" in the paper describes the roles of the authors and the models and the validation performed. Two independent blind audits read the manuscript; no formal verification is claimed.

## Status, rights and citation

Text and documentation: CC BY 4.0. Code: MIT. See [RIGHTS.md](RIGHTS.md).
