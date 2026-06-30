"""
TractoPersist - Run All Figures
Master script to generate all paper figures
in correct order

Run: python step_run_all_figures.py
"""

import os
import sys
import subprocess
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

scripts_dir = r"D:\CHINA LAPTOP\new paper"
fig_path    = r"F:\ADNI_Features\figures_paper"

scripts = [
    "step_fig5_pipeline.py",
    "step_fig1_brain_maps.py",
    "step_fig2_connectome.py",
    "step_fig3_statistics.py",
    "step_fig4_correlation.py",
    "step_fig6_persistence.py",
]

print("=" * 60)
print("  TractoPersist — Generate All Figures")
print("=" * 60)
print(f"  Output: {fig_path}\n")

for i, script in enumerate(scripts):
    script_path = os.path.join(
        scripts_dir, script)
    print(f"  [{i+1}/{len(scripts)}] "
          f"Running: {script}")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300)
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"    ✓ Done in "
                  f"{elapsed:.0f}s")
        else:
            print(f"    ✗ Error:")
            print(f"      "
                  f"{result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout!")
    except Exception as e:
        print(f"    ✗ {e}")

print("\n" + "=" * 60)
print("  COMPLETE!")
print("=" * 60)
print(f"\n  Check: {fig_path}")
print("\n  Files generated:")
expected = [
    "fig1_brain_maps.png",
    "fig2_connectome.png",
    "fig3_statistics.png",
    "fig4_correlation.png",
    "fig5_pipeline.png",
    "fig6_persistence.png",
]
for f in expected:
    fp = os.path.join(fig_path, f)
    exists = "✓" if os.path.exists(fp) \
        else "✗ MISSING"
    sz = f"({os.path.getsize(fp)//1024} KB)" \
        if os.path.exists(fp) else ""
    print(f"  {exists}  {f} {sz}")
