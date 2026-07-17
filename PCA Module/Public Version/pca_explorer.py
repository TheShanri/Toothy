#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Explorer (public / portable version)
=========================================

Interactive viewer for Toothy dentate-spike PCA classification.

  • Pick a **Toothy folder** (the one containing `CSDs.hdf5`) — the app reads the
    saved DS classification (pc1, pc2, type, ...) for each dentate-spike event.
  • Pick an **image folder** — the per-event visualization images (any common
    image type). Images are matched to events in chronological order.
  • The PCA scatter is drawn, one dot per event, colored by DS type.
  • Click a dot to load that event's image on the right.
  • The image panel supports zoom (mouse wheel or buttons) and pan (drag).

Nothing here depends on the raw .ncs data or Toothy's CSD/quantities stack —
the PCA coordinates are read straight from the saved CSDs.hdf5.

Run:
    python pca_explorer.py

Requires: PyQt5, matplotlib, pandas, numpy, h5py, pytables (tables).

Author: Shahriar Tafti, UVM
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.lines
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# DS-type colors (Toothy-style blue/red split); 0 = unclassified / DBSCAN noise.
TYPE_COLORS = {0: "#9e9e9e", 1: "#1f77d4", 2: "#d62728"}
TYPE_LABELS = {0: "Unclassified", 1: "DS Type 1", 2: "DS Type 2"}
IMAGE_EXTS  = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")


# ─────────────────────────────────────────────────────────────────────────────
#  Zoomable / pannable image viewer
# ─────────────────────────────────────────────────────────────────────────────
class ImageViewer(QtWidgets.QGraphicsView):
    """QGraphicsView showing one image, with wheel-zoom and drag-pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#111111")))
        self.setMinimumSize(360, 360)
        self._zoom = 0
        self._empty = True

    def has_photo(self):
        return not self._empty

    def set_photo(self, pixmap=None):
        self._zoom = 0
        if pixmap is not None and not pixmap.isNull():
            self._empty = False
            self._item.setPixmap(pixmap)
            self.fit()
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._item.setPixmap(QtGui.QPixmap())

    def fit(self):
        rect = QtCore.QRectF(self._item.pixmap().rect())
        if rect.isNull():
            return
        self.setSceneRect(rect)
        self.resetTransform()
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self._zoom = 0

    def _apply_zoom(self, step):
        if self._empty:
            return
        factor = 1.25 if step > 0 else 0.8
        self._zoom += step
        if self._zoom <= 0:
            self.fit()
        else:
            self.scale(factor, factor)

    def zoom_in(self):
        self._apply_zoom(+1)

    def zoom_out(self):
        self._apply_zoom(-1)

    def wheelEvent(self, event):
        self._apply_zoom(+1 if event.angleDelta().y() > 0 else -1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom == 0 and not self._empty:
            self.fit()


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────
class PCAExplorer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QtCore.QSettings("Toothy", "PCAExplorer")
        self.df = pd.DataFrame()
        self.images = []
        self.image_garbage = []
        self.color_col = "type"
        self.sel = None
        self._dsdf_keys = []          # available (probe,shank) DS_DF keys

        self.setWindowTitle("Toothy PCA Explorer")
        self.resize(1320, 820)
        self._build_ui()
        self._restore_paths()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # --- top: folder pickers ------------------------------------------------
        top = QtWidgets.QGridLayout()
        self.toothy_le = QtWidgets.QLineEdit()
        self.toothy_le.setPlaceholderText("Toothy folder (contains CSDs.hdf5)…")
        b1 = QtWidgets.QPushButton("Browse…"); b1.clicked.connect(self._pick_toothy)
        self.img_le = QtWidgets.QLineEdit()
        self.img_le.setPlaceholderText("Image folder (per-event visualizations)…")
        b2 = QtWidgets.QPushButton("Browse…"); b2.clicked.connect(self._pick_images)

        self.shank_combo = QtWidgets.QComboBox()
        self.shank_combo.setMinimumWidth(150)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setStyleSheet("font-weight:bold;")
        self.load_btn.clicked.connect(self._load)

        top.addWidget(QtWidgets.QLabel("Toothy:"), 0, 0)
        top.addWidget(self.toothy_le, 0, 1)
        top.addWidget(b1, 0, 2)
        top.addWidget(QtWidgets.QLabel("Probe/Shank:"), 0, 3)
        top.addWidget(self.shank_combo, 0, 4)
        top.addWidget(QtWidgets.QLabel("Images:"), 1, 0)
        top.addWidget(self.img_le, 1, 1)
        top.addWidget(b2, 1, 2)
        top.addWidget(self.load_btn, 1, 3, 1, 2)
        top.setColumnStretch(1, 1)
        outer.addLayout(top)

        # --- main split: scatter | image ---------------------------------------
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # left panel
        leftw = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(leftw)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(QtWidgets.QLabel("Color by:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(["Final type", "KMeans (k_type)", "DBSCAN (db_type)"])
        self.color_combo.currentIndexChanged.connect(self._on_color_changed)
        crow.addWidget(self.color_combo)
        self.gray_chk = QtWidgets.QCheckBox("Gray out garbage")
        self.gray_chk.setToolTip("Gray out events whose image is in the 'Garbage' subfolder")
        self.gray_chk.toggled.connect(lambda _: self._draw_scatter())
        crow.addWidget(self.gray_chk)
        crow.addStretch(1)
        self.count_lbl = QtWidgets.QLabel("")
        crow.addWidget(self.count_lbl)
        left.addLayout(crow)

        self.fig = Figure(figsize=(6, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        left.addWidget(NavigationToolbar(self.canvas, self))
        left.addWidget(self.canvas, 1)
        split.addWidget(leftw)

        # right panel
        rightw = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(rightw)
        self.info_lbl = QtWidgets.QLabel("Load a Toothy folder and image folder to begin.")
        self.info_lbl.setStyleSheet("font-weight:bold; font-size:13px;")
        self.info_lbl.setWordWrap(True)
        right.addWidget(self.info_lbl)

        self.viewer = ImageViewer()
        right.addWidget(self.viewer, 1)

        zrow = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        zoom_out = QtWidgets.QPushButton("–")
        zoom_in = QtWidgets.QPushButton("+")
        fit_btn = QtWidgets.QPushButton("Fit")
        for b in (zoom_out, zoom_in, fit_btn):
            b.setMaximumWidth(48)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        zoom_out.clicked.connect(self.viewer.zoom_out)
        zoom_in.clicked.connect(self.viewer.zoom_in)
        fit_btn.clicked.connect(self.viewer.fit)
        zrow.addWidget(self.prev_btn)
        zrow.addWidget(self.next_btn)
        zrow.addStretch(1)
        zrow.addWidget(QtWidgets.QLabel("Zoom:"))
        zrow.addWidget(zoom_out)
        zrow.addWidget(zoom_in)
        zrow.addWidget(fit_btn)
        right.addLayout(zrow)
        split.addWidget(rightw)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)

        self.status = self.statusBar()
        self.status.showMessage("Ready.")

    # ── path handling ────────────────────────────────────────────────────────
    def _restore_paths(self):
        t = self.settings.value("toothy_dir", "", type=str)
        i = self.settings.value("image_dir", "", type=str)
        if t:
            self.toothy_le.setText(t); self._scan_shanks(t)
        if i:
            self.img_le.setText(i)

    def _pick_toothy(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Toothy folder (contains CSDs.hdf5)", self.toothy_le.text())
        if d:
            self.toothy_le.setText(d)
            self._scan_shanks(d)

    def _pick_images(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select image folder", self.img_le.text())
        if d:
            self.img_le.setText(d)

    def _csds_path(self, toothy_dir):
        p = Path(toothy_dir)
        if p.is_file() and p.suffix == ".hdf5":
            return p
        cand = p / "CSDs.hdf5"
        return cand if cand.exists() else None

    def _scan_shanks(self, toothy_dir):
        """Populate the probe/shank dropdown from CSDs.hdf5 groups holding a DS_DF."""
        self.shank_combo.clear()
        self._dsdf_keys = []
        csds = self._csds_path(toothy_dir)
        if not csds:
            self.status.showMessage("No CSDs.hdf5 found in that folder.")
            return
        keys = []
        try:
            with h5py.File(str(csds), "r") as F:
                for probe in sorted(F.keys()):
                    g = F[probe]
                    if not isinstance(g, h5py.Group):
                        continue
                    for shank in sorted(g.keys()):
                        sg = g[shank]
                        if isinstance(sg, h5py.Group) and "DS_DF" in sg:
                            keys.append((probe, shank))
        except Exception as e:
            self.status.showMessage(f"Could not read CSDs.hdf5: {e}")
            return
        self._dsdf_keys = keys
        for probe, shank in keys:
            self.shank_combo.addItem(f"probe {probe} / shank {shank}")
        self.status.showMessage(f"Found {len(keys)} classified shank(s) in CSDs.hdf5.")

    # ── loading ───────────────────────────────────────────────────────────────
    def _read_lfp_fs(self, toothy_dir):
        data = Path(toothy_dir) / "DATA.hdf5"
        if data.exists():
            try:
                with h5py.File(str(data), "r") as F:
                    return float(F.attrs.get("lfp_fs", 0)) or None
            except Exception:
                pass
        return None

    @staticmethod
    def _img_sort_key(f):
        name = f.stem
        m = re.search(r"evt[_\-\s]?(\d+)", name, re.I)
        if not m:
            m = re.search(r"(\d+)", name)
        return (0, int(m.group(1))) if m else (1, name.lower())

    def _scan_images(self, image_dir):
        """Collect event images from `image_dir`, folding in any 'Garbage'
        subfolder. Images are keyed by filename so a copy moved into Garbage still
        maps to the right event; a file present in the Garbage subfolder is flagged
        garbage (and wins over a same-named file up top). Sorted chronologically by
        the number in the filename.

        Returns (paths, garbage_flags) as two aligned lists.
        """
        p = Path(image_dir)
        entries = {}   # filename.lower() -> {"path": Path, "garbage": bool}
        if p.is_dir():
            for f in p.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    entries[f.name.lower()] = {"path": f, "garbage": False}
            for sub in p.iterdir():
                if sub.is_dir() and sub.name.lower() == "garbage":
                    for f in sub.iterdir():
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                            entries[f.name.lower()] = {"path": f, "garbage": True}
        items = sorted(entries.values(), key=lambda d: self._img_sort_key(d["path"]))
        return [d["path"] for d in items], [d["garbage"] for d in items]

    def _load(self):
        toothy_dir = self.toothy_le.text().strip()
        image_dir = self.img_le.text().strip()
        if not toothy_dir:
            QtWidgets.QMessageBox.warning(self, "Missing folder", "Select a Toothy folder.")
            return
        csds = self._csds_path(toothy_dir)
        if not csds:
            QtWidgets.QMessageBox.warning(self, "Not found",
                "No CSDs.hdf5 in that folder. This recording may not be classified yet.")
            return
        if self.shank_combo.count() == 0:
            self._scan_shanks(toothy_dir)
        if not self._dsdf_keys:
            QtWidgets.QMessageBox.warning(self, "No classification",
                "CSDs.hdf5 has no saved DS_DF (no PCA classification stored).")
            return

        probe, shank = self._dsdf_keys[max(0, self.shank_combo.currentIndex())]
        key = f"/{probe}/{shank}/DS_DF"
        try:
            df = pd.read_hdf(str(csds), key=key)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Read error", f"Could not read {key}:\n{e}")
            return

        if "idx" in df.columns:
            df = df.sort_values("idx", kind="stable")
        df = df.reset_index(drop=True)
        for c in ("pc1", "pc2"):
            if c not in df.columns:
                QtWidgets.QMessageBox.critical(self, "No PCA",
                    f"'{c}' missing from DS_DF — no PCA coordinates to plot.")
                return
        for c in ("type", "k_type", "db_type"):
            if c not in df.columns:
                df[c] = df.get("type", 1)

        fs = self._read_lfp_fs(toothy_dir)
        df["time_sec"] = (df["idx"].astype(float) / fs) if (fs and "idx" in df) else np.nan
        df["event"] = np.arange(1, len(df) + 1)

        self.images, self.image_garbage = self._scan_images(image_dir)
        garbage_col = np.zeros(len(df), dtype=bool)
        for i in range(min(len(df), len(self.image_garbage))):
            garbage_col[i] = self.image_garbage[i]
        df["garbage"] = garbage_col
        self.df = df
        self.color_col = ["type", "k_type", "db_type"][self.color_combo.currentIndex()]

        # persist paths
        self.settings.setValue("toothy_dir", toothy_dir)
        self.settings.setValue("image_dir", image_dir)

        n_ev, n_img = len(df), len(self.images)
        n_garbage = int(df["garbage"].sum())
        self.gray_chk.setText(f"Gray out garbage ({n_garbage})")
        self.gray_chk.setEnabled(n_garbage > 0)
        msg = f"Loaded {n_ev} events (probe {probe}/shank {shank}); {n_img} images"
        msg += f"; {n_garbage} in Garbage." if n_garbage else "."
        if n_img and n_img != n_ev:
            msg += f"  ⚠ event/image count mismatch ({n_ev} vs {n_img}) — matched by order."
        elif n_img == 0:
            msg += "  ⚠ no images found in image folder."
        self.status.showMessage(msg)

        self._draw_scatter()
        self._select(0)

    # ── scatter ───────────────────────────────────────────────────────────────
    def _draw_scatter(self):
        self.ax.clear()
        if not len(self.df):
            self.canvas.draw_idle()
            return
        vals = self.df[self.color_col].astype(int).values
        garbage = (self.df["garbage"].values if "garbage" in self.df.columns
                   else np.zeros(len(vals), dtype=bool))
        gray_mask = garbage & self.gray_chk.isChecked()

        rgba = np.array([mcolors.to_rgba("#c8c8c8", 0.22) if g
                         else mcolors.to_rgba(TYPE_COLORS.get(v, "#9e9e9e"), 0.85)
                         for v, g in zip(vals, gray_mask)])
        self.ax.scatter(self.df["pc1"], self.df["pc2"], c=rgba, s=42,
                        edgecolors="white", linewidths=0.4, zorder=2)

        # legend: type counts among non-grayed points, plus a Garbage entry
        handles = []
        for v in sorted(set(vals[~gray_mask])):
            n = int(((vals == v) & ~gray_mask).sum())
            handles.append(matplotlib.lines.Line2D([], [], marker="o", linestyle="",
                markerfacecolor=TYPE_COLORS.get(v, "#9e9e9e"), markeredgecolor="white",
                markersize=8, label=f"{TYPE_LABELS.get(v, v)}  (n={n})"))
        if gray_mask.any():
            handles.append(matplotlib.lines.Line2D([], [], marker="o", linestyle="",
                markerfacecolor="#c8c8c8", markeredgecolor="white", markersize=8,
                label=f"Garbage  (n={int(gray_mask.sum())})"))
        self.ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
        self.ax.set_xlabel("Principal Component 1")
        self.ax.set_ylabel("Principal Component 2")
        self.ax.set_title("PCA of Dentate-Spike CSD profiles\n(click a point)", fontweight="bold")
        self.ax.grid(True, alpha=0.25)
        (self._sel_marker,) = self.ax.plot([], [], "o", markersize=16, markerfacecolor="none",
                                           markeredgecolor="#111", markeredgewidth=2, zorder=3)
        self.count_lbl.setText(f"{len(self.df)} events")
        if self.sel is not None and self.sel < len(self.df):
            self._update_marker()
        self.canvas.draw_idle()

    def _update_marker(self):
        if self.sel is None:
            return
        row = self.df.iloc[self.sel]
        self._sel_marker.set_data([row["pc1"]], [row["pc2"]])
        self.canvas.draw_idle()

    # ── interaction ─────────────────────────────────────────────────────────
    def _on_color_changed(self, i):
        self.color_col = ["type", "k_type", "db_type"][i]
        if len(self.df):
            self._draw_scatter()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or not len(self.df):
            return
        xy = self.ax.transData.transform(self.df[["pc1", "pc2"]].values)
        click = np.array([event.x, event.y])
        d = np.hypot(xy[:, 0] - click[0], xy[:, 1] - click[1])
        i = int(np.argmin(d))
        if d[i] > 40:
            return
        self._select(i)

    def _step(self, delta):
        if self.sel is None or not len(self.df):
            return
        self._select(int(np.clip(self.sel + delta, 0, len(self.df) - 1)))

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Left:
            self._step(-1)
        elif e.key() == QtCore.Qt.Key_Right:
            self._step(+1)
        else:
            super().keyPressEvent(e)

    def _select(self, i):
        if not len(self.df):
            return
        self.sel = i
        row = self.df.iloc[i]
        self._update_marker()
        t = row.get("time_sec", np.nan)
        tstr = f"{t:.3f} s" if pd.notna(t) else "n/a"
        idx = int(row["idx"]) if "idx" in row and pd.notna(row["idx"]) else "n/a"
        gtag = "   ⛔ GARBAGE" if bool(row.get("garbage", False)) else ""
        self.info_lbl.setText(
            f"Event #{int(row['event'])} / {len(self.df)}    "
            f"[{TYPE_LABELS.get(int(row['type']), row['type'])}]{gtag}\n"
            f"t = {tstr}   (sample idx {idx})\n"
            f"PC1 = {row['pc1']:.3f}   PC2 = {row['pc2']:.3f}\n"
            f"KMeans={int(row['k_type'])}  DBSCAN={int(row['db_type'])}")
        self._show_image(i)

    def _show_image(self, i):
        if i < len(self.images):
            path = self.images[i]
            pm = QtGui.QPixmap(str(path))
            if not pm.isNull():
                self.viewer.set_photo(pm)
                self.setWindowTitle(f"Toothy PCA Explorer — {path.name}")
                return
        self.viewer.set_photo(None)
        self.setWindowTitle("Toothy PCA Explorer")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PCAExplorer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
