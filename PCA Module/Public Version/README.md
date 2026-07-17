# Toothy PCA Explorer (portable)

Interactive viewer for Toothy's dentate-spike PCA classification. Pick a Toothy
folder and an image folder, and click points in the PCA scatter to view each
event's visualization.

## Run

```bash
python pca_explorer.py
```

Use a Python with: **PyQt5, matplotlib, pandas, numpy, h5py, tables (pytables)**.
On this machine that is:

```bash
"C:/Users/Z390/anaconda3/python.exe" pca_explorer.py
```

(Do **not** use `toothy_env` — its sklearn install crashes. This app doesn't need
sklearn: the PCA coordinates are read straight from the saved `CSDs.hdf5`.)

## How to use

1. **Toothy:** browse to the recording's Toothy folder — the one containing
   `CSDs.hdf5` (you may also point directly at a `CSDs.hdf5` file). The
   **Probe/Shank** dropdown fills with every shank that has a saved
   classification.
2. **Images:** browse to the folder of per-event images (PNG/JPG/TIF/…).
3. Click **Load**.

Images are matched to events **in chronological order** (event #1 = earliest
dentate spike = first image). Image files are ordered by the number in their
filename (e.g. `..._Evt007_...` → 7), falling back to natural name order. If the
image count differs from the event count, the status bar warns you and matches as
far as they go.

## Controls

- **Click a dot** → load that event's image.
- **← / →** → step to previous / next event.
- **Color by** → recolor dots by final type, KMeans, or DBSCAN label.
- **Gray out garbage** → see below.
- **Image panel:** mouse-wheel or **+ / –** to zoom, drag to pan, **Fit** to reset.

## Garbage folder

If the image folder contains a subfolder named **`Garbage`** (case-insensitive),
any event whose image lives in it is flagged as garbage. Images are matched to
events by the number in the filename, so an image *moved* into `Garbage` still
maps to the correct event (it does not shift the ordering of the others).

The **"Gray out garbage (N)"** checkbox (top-left, next to *Color by*) renders
those events as faint gray and pulls them out of the per-type legend counts;
unchecking restores their normal DS-type colors. Selected garbage events are
tagged **⛔ GARBAGE** in the info panel. The checkbox is disabled when no
`Garbage` folder / garbage images are present.

## Notes

- The dot coordinates and DS-type colors are Toothy's own saved values
  (`/<probe>/<shank>/DS_DF` in `CSDs.hdf5`); nothing is recomputed.
- Event time (seconds) is shown when a sibling `DATA.hdf5` provides `lfp_fs`;
  otherwise only the sample index is shown.
- Selected folders are remembered between sessions.
