#!/bin/bash
# ================================================================
# TractoPersist — MRtrix3 Preprocessing Pipeline
# Runs on Ubuntu WSL2
# Usage: bash pipeline.sh
# ================================================================

ATLAS="/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
MRTRIX="/mnt/e/WSL/mrtrix3/bin"
export PATH="$MRTRIX:$PATH"

echo "=== TractoPersist Pipeline ==="
echo "Started: $(date)"

for GROUP in AD_v2 MCI_v2 CN1; do
    INPUT="/mnt/f/ADNI_NIfTI/$GROUP"
    echo ""
    echo ">>> Group: $GROUP"

    for subject_dir in $INPUT/*/; do
        subject=$(basename $subject_dir)

        DWI="${subject_dir}/${subject}.nii.gz"
        BVEC="${subject_dir}/${subject}.bvec"
        BVAL="${subject_dir}/${subject}.bval"

        [ ! -f "$DWI" ] && \
            echo "  SKIP (no DWI): $subject" && continue

        echo "  Processing: $subject"

        # Step 1: Denoise
        DEN="${subject_dir}/${subject}_denoised.nii.gz"
        [ ! -f "$DEN" ] && \
        dwidenoise $DWI $DEN \
            -fslgrad $BVEC $BVAL -force

        # Step 2: Gibbs
        GIB="${subject_dir}/${subject}_gibbs.nii.gz"
        [ ! -f "$GIB" ] && \
        mrdegibbs $DEN $GIB -force

        # Step 3: Brain mask
        MASK="${subject_dir}/${subject}_mask.nii.gz"
        [ ! -f "$MASK" ] && \
        dwi2mask $GIB $MASK \
            -fslgrad $BVEC $BVAL -force

        # Step 4: Response function
        RESP="${subject_dir}/${subject}_response.txt"
        [ ! -f "$RESP" ] && \
        dwi2response tournier $GIB $RESP \
            -fslgrad $BVEC $BVAL \
            -mask $MASK -force

        # Step 5: FOD
        FOD="${subject_dir}/${subject}_fod.nii.gz"
        [ ! -f "$FOD" ] && \
        dwi2fod csd $GIB $RESP $FOD \
            -fslgrad $BVEC $BVAL \
            -mask $MASK -force

        # Step 6: Tractography
        TCK="${subject_dir}/${subject}_tracks.tck"
        [ ! -f "$TCK" ] && \
        tckgen $FOD $TCK \
            -algorithm iFOD2 \
            -select 100000 \
            -seed_image $MASK \
            -mask $MASK -force

        # Step 7: SIFT2
        SIFT="${subject_dir}/${subject}_sift2weights.txt"
        [ ! -f "$SIFT" ] && \
        tcksift2 $TCK $FOD $SIFT -force

        # Step 8: Connectome
        CONN="${subject_dir}/${subject}_connectome.csv"
        [ ! -f "$CONN" ] && \
        tck2connectome $TCK $ATLAS $CONN \
            -tck_weights_in $SIFT \
            -symmetric -zero_diagonal \
            -force

        echo "  DONE: $subject"
    done
done

echo ""
echo "=== Pipeline Complete: $(date) ==="
