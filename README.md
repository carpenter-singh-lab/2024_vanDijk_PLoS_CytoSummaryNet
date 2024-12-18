# CytoSummaryNet

This repository implements `CytoSummaryNet` the method described in "Capturing cell heterogeneity in representations of cell populations for image-based profiling using contrastive learning" ([van Dijk et al., 2024, PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1012547)).

`CytoSummaryNet` learns an optimal way to aggregate single-cell features into population-level profiles, outperforming traditional averaging on tasks like mechanism-of-action prediction.

## Repository Structure

```
.
├── cytosummarynet/     # Core package implementation
└── paper_experiments/  # Code to reproduce paper results
```

## Quick Start

Install the package:
```bash
pip install cytosummarynet
```

For detailed documentation:
- Core package usage: See `cytosummarynet/README.md`
- Reproducing paper results: See `paper_experiments/README.md`

## Citation

If you use this code, please cite:

```
van Dijk, R., Arevalo, J., Babadi, B., Carpenter, A. E., & Singh, S. (2024).
Capturing cell heterogeneity in representations of cell populations for image-based profiling using contrastive learning. PLOS Computational Biology.
```