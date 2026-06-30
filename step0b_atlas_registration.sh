#!/bin/bash
# ================================================================
# TractoPersist — Atlas Registration to Native DWI Space
# Registers the Harvard-Oxford 48-ROI cortical atlas (MNI space)
# into each subject's native diffusion space, producing
# <subject>_atlas.nii.gz, which step3_combine_features.py
# uses to compute per-ROI FA/MD/RD means.
#
# Requires: FSL (flirt, fnirt or applywarp) on PATH.
# Usage: bash step0b_atlas_registration.sh
# ================================================================

MRTRIX="/mnt/e/WSL/mrtrix3/bin"
export PATH="$MRTRIX:$PATH"

ATLAS_MNI="/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
MNI_TEMPLATE="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"

echo "=== TractoPersist: Atlas Registration ==="
echo "Started: $(date)"

for GROUP in AD_v2 MCI_v2 CN1; do
    INPUT="/mnt/f/ADNI_NIfTI/$GROUP"
    echo ""
    echo ">>> Group: $GROUP"

    for subject_dir in $INPUT/*/; do
        subject=$(basename "$subject_dir")

        FA="${subject_dir}/${subject}_FA.nii.gz"
        MASK="${subject_dir}/${subject}_mask.nii.gz"
        ATLAS_OUT="${subject_dir}/${subject}_atlas.nii.gz"

        [ ! -f "$FA" ] && \
            echo "  SKIP (no FA, run extract_metrics.sh first): $subject" && continue
        [ -f "$ATLAS_OUT" ] && \
            echo "  SKIP (atlas exists): $subject" && continue

        echo "  Registering atlas: $subject"

        XFM="${subject_dir}/${subject}_mni2dwi.mat"
        FA_BRAIN="${subject_dir}/${subject}_FA_brain.nii.gz"

        # Mask FA for registration target
        fslmaths "$FA" -mas "$MASK" "$FA_BRAIN"

        # Affine registration: MNI brain -> subject FA
        flirt -in "$MNI_TEMPLATE" \
              -ref "$FA_BRAIN" \
              -omat "$XFM" \
              -dof 12 \
              -cost corratio \
              -searchrx -30 30 -searchry -30 30 -searchrz -30 30

        # Apply transform to atlas labels (nearest-neighbour
        # to preserve discrete ROI labels)
        flirt -in "$ATLAS_MNI" \
              -ref "$FA_BRAIN" \
              -applyxfm -init "$XFM" \
              -interp nearestneighbour \
              -out "$ATLAS_OUT"

        echo "  DONE: $subject"
    done
done

echo ""
echo "=== Atlas Registration Complete: $(date) ==="
echo ""
echo "QC TIP: overlay the registered atlas on FA to check alignment:"
echo "  mrview <subject>_FA.nii.gz -overlay.load <subject>_atlas.nii.gz"
