#!/bin/bash
set -e

OUTPUT_DIR="data/almost_all_cells_negative"
mkdir -p "${OUTPUT_DIR}"

sc-big simulate \
  -n 500 \
  --ccf 0.20 \
  --purity 1.0 \
  --copy-number 2 \
  --multiplicity 1 \
  --sc-coverage 3 \
  --bulk-coverage 250 \
  --epsilon-sc 0.02 \
  --n-germline-snps 600 \
  --base-concentration-mean 80.0 \
  --base-concentration-cv 0.05 \
  --seed 42 \
  -o "${OUTPUT_DIR}/simulation.json" \
  -p "${OUTPUT_DIR}/simulation_plots.png" \
  -v

sc-big infer \
  -d "${OUTPUT_DIR}/simulation.json" \
  --purity-mean 1.0 \
  --purity-std 0.01 \
  --copy-number-mean 2.0 \
  --copy-number-std 0.3 \
  --epsilon-sc 0.02 \
  --multiplicity-prior geometric \
  --n-workers 15 \
  -o "${OUTPUT_DIR}/bayesian_results.json" \
  -p "${OUTPUT_DIR}/bayesian_plots.png" \
  --diagnostics \
  -v

echo "All results saved to: ${OUTPUT_DIR}/"
