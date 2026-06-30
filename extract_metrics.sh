#!/bin/bash
# ================================================================
# TractoPersist — DTI Metrics Extraction
# Extracts FA, MD, RD maps per subject
# Usage: bash extract_metrics.sh
# ================================================================

MRTRIX="/mnt/e/WSL/mrtrix3/bin"
export PATH="$MRTRIX:$PATH"

echo "=== DTI Metrics Extraction ==="

for GROUP in AD_v2 MCI_v2 CN1; do
    INPUT="/mnt/f/ADNI_NIfTI/$GROUP"
    echo "Group: $GROUP"

    for subject_dir in $INPUT/*/; do
        subject=$(basename $subject_dir)

        GIB="${subject_dir}/${subject}_gibbs.nii.gz"
        BVEC="${subject_dir}/${subject}.bvec"
        BVAL="${subject_dir}/${subject}.bval"
        MASK="${subject_dir}/${subject}_mask.nii.gz"

        [ ! -f "$GIB" ] && continue

        TENSOR="${subject_dir}/${subject}_tensor.nii.gz"
        FA="${subject_dir}/${subject}_FA.nii.gz"
        MD="${subject_dir}/${subject}_MD.nii.gz"
        RD="${subject_dir}/${subject}_RD.nii.gz"

        if [ ! -f "$FA" ]; then
            echo "  Metrics: $subject"
            dwi2tensor $GIB $TENSOR \
                -fslgrad $BVEC $BVAL \
                -mask $MASK -force

            tensor2metric $TENSOR \
                -fa $FA -md $MD \
                -rd $RD -force

            rm -f $TENSOR
        fi
    done
done
echo "=== Done ==="
