#!/bin/bash
# ================================================================
# TractoPersist — Brain Masking (standalone)
# Extracts a robust brain mask from the b0 volume for each
# subject. This is also performed inline inside pipeline.sh,
# but is provided here standalone for:
#   (a) re-running/QC on masking alone without re-running
#       denoise/Gibbs/CSD/tractography, and
#   (b) visual inspection of mask quality before committing
#       to the full pipeline.
#
# Usage: bash step0_brain_masking.sh
# ================================================================

MRTRIX="/mnt/e/WSL/mrtrix3/bin"
export PATH="$MRTRIX:$PATH"

echo "=== TractoPersist: Brain Masking ==="
echo "Started: $(date)"

for GROUP in AD_v2 MCI_v2 CN1; do
    INPUT="/mnt/f/ADNI_NIfTI/$GROUP"
    echo ""
    echo ">>> Group: $GROUP"

    for subject_dir in $INPUT/*/; do
        subject=$(basename "$subject_dir")

        DWI="${subject_dir}/${subject}.nii.gz"
        BVEC="${subject_dir}/${subject}.bvec"
        BVAL="${subject_dir}/${subject}.bval"

        [ ! -f "$DWI" ] && \
            echo "  SKIP (no DWI): $subject" && continue

        MASK="${subject_dir}/${subject}_mask.nii.gz"
        B0="${subject_dir}/${subject}_b0.nii.gz"

        if [ -f "$MASK" ]; then
            echo "  SKIP (mask exists): $subject"
            continue
        fi

        echo "  Masking: $subject"

        # Extract mean b0 volume for QC visualisation
        dwiextract "$DWI" "$B0" \
            -fslgrad "$BVEC" "$BVAL" \
            -bzero -force

        # Robust brain mask (MRtrix3 dwi2mask)
        dwi2mask "$DWI" "$MASK" \
            -fslgrad "$BVEC" "$BVAL" -force

        # Mask volume sanity check (voxel count)
        VOXELS=$(mrstats "$MASK" -mask "$MASK" \
            -output count 2>/dev/null)
        echo "    Mask voxels: $VOXELS"

        echo "  DONE: $subject"
    done
done

echo ""
echo "=== Brain Masking Complete: $(date) ==="
echo ""
echo "QC TIP: visually inspect a few masks with:"
echo "  mrview <subject>_b0.nii.gz -overlay.load <subject>_mask.nii.gz"
