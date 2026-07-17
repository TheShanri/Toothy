#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_pca_points.py

Build a portable `pca_points.csv` that the PCA Explorer GUI reads, so the GUI
never has to touch the raw D: drive or any CSD dependency at runtime.

Source of truth: Toothy's saved GUI classification for the M2s8jan26 recording:
    <TOOTHY_DIR>/CSDs.hdf5   ->   /0/0/DS_DF   (209 dentate-spike events)

That DS_DF already contains the exact PCA coordinates Toothy computed and saved
(pc1, pc2) plus the cluster labels (k_type, db_type, type) and the sample index
(idx). Events are stored in ascending idx == chronological order, which is the
same order the raster images are numbered (Raster_Evt001.._1ch.png).

We simply copy those columns out, attach the matching image filename, and write
a CSV. No recomputation — these are Toothy's own values. (pca_module.run_pca was
independently verified to reproduce pc1/pc2 from the saved norm_filt_csd to
~1e-15, so the pipeline is faithful.)

Author: Shahriar Tafti, UVM
"""

from pathlib import Path
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
TOOTHY_DIR = Path(r"D:/PTEN/CTL/M2_CTL/M2s8jan26/2024-01-26_17-48-00/toothy")
CSDS_H5    = TOOTHY_DIR / "CSDs.hdf5"
DS_DF_KEY  = "/0/0/DS_DF"          # probe 0, shank 0
LFP_FS     = 1000.0               # Hz (DATA.hdf5 attr for this recording)

HERE       = Path(__file__).resolve().parent
IMAGE_DIR  = HERE / "Test Data" / "M2s8jan26"
IMAGE_TMPL = "Raster_Evt{n:03d}_1ch.png"   # n = 1-based chronological event #
OUT_CSV    = HERE / "pca_points.csv"


def main():
    df = pd.read_hdf(CSDS_H5, key=DS_DF_KEY).reset_index(drop=True)

    # events are already chronological (ascending idx); enforce it to be safe
    df = df.sort_values("idx", kind="stable").reset_index(drop=True)

    n = len(df)
    out = pd.DataFrame({
        "event"    : range(1, n + 1),                 # 1-based chronological #
        "idx"      : df["idx"].astype(int),           # LFP sample index
        "time_sec" : df["idx"].astype(float) / LFP_FS,
        "pc1"      : df["pc1"].astype(float),
        "pc2"      : df["pc2"].astype(float),
        "type"     : df["type"].astype(int),          # saved final DS type (1/2)
        "k_type"   : df["k_type"].astype(int),        # KMeans label
        "db_type"  : df["db_type"].astype(int),       # DBSCAN label (0 = noise)
    })
    out["image"] = [IMAGE_TMPL.format(n=i) for i in out["event"]]

    # flag which images actually exist on disk (non-fatal)
    out["image_exists"] = [(IMAGE_DIR / img).exists() for img in out["image"]]

    out.to_csv(OUT_CSV, index=False)

    missing = int((~out["image_exists"]).sum())
    print(f"Wrote {OUT_CSV}  ({n} events)")
    print(f"  type counts : {out['type'].value_counts().sort_index().to_dict()}")
    print(f"  images found: {n - missing}/{n}"
          + ("" if missing == 0 else f"   (MISSING {missing})"))
    print(f"  image dir   : {IMAGE_DIR}")


if __name__ == "__main__":
    main()
