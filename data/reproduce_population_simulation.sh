#!/bin/bash
set -e

command -v prosolo >/dev/null 2>&1 || {
    echo "ERROR: 'prosolo' not found on PATH." >&2
    echo "Activate the prosolo conda environment first." >&2
    exit 1
}

OUTPUT_DIR="/scratch/dschuet7/population_simulation_$(date +%Y%m%d_%H%M%S)"
NUM_WORKERS=124
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

sc-big simulate-tree -n 1000 --n-mutations 200 \
  --ccf-distribution uniform \
  --purity 1.0 --sc-coverage 3 --bulk-coverage 250 \
  --epsilon-sc 0.01 --n-germline-snps 10 \
  --error-model lodato --seed 42 \
  -o "${OUTPUT_DIR}" -p "${OUTPUT_DIR}/simulation.png" -v

for f in "${OUTPUT_DIR}"/mutation_[0-9]*.json; do
  # Skip result files (sc-big / prosolo outputs).
  case "$f" in *_scbig.json|*_prosolo*.json) continue ;; esac

  # Parse copy number from filename (e.g. mutation_0003_C3_m2_...).
  C=$(echo "$f" | grep -oP '(?<=_C)\d+')

  sc-big infer -d "$f" --error-model lodato \
    --purity-mean 1.0 --purity-std 0.01 \
    --copy-number-mean "$C" --copy-number-std 0.3 --epsilon-sc 0.01 \
    --multiplicity-prior geometric --n-workers "${NUM_WORKERS}" \
    -o "${f%.json}_scbig.json" -p "${f%.json}_scbig.png" --diagnostics -v

  sc-big infer-prosolo -d "$f" --base-error-rate 0.01 \
    --n-workers "${NUM_WORKERS}" -o "${f%.json}_prosolo_native.json"
done

sc-big compare \
  --scbig "${OUTPUT_DIR}"/mutation_*_scbig.json \
  --prosolo "${OUTPUT_DIR}"/mutation_*_prosolo_native.json \
  -o "${OUTPUT_DIR}/comparison.png"

echo "All results saved to: ${OUTPUT_DIR}/"
