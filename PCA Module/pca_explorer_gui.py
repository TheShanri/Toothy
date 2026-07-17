#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Explorer GUI
================

Interactive scatter of the dentate-spike PCA (one dot per DS event). Click a dot
to load that event's raster visualization (the "Test Data" image) beside the
plot. Dots are colored by their saved DS type (Type 1 vs Type 2).

Data source: `pca_points.csv` produced by prepare_pca_points.py, which pulls the
exact pc1/pc2/type Toothy saved for the M2s8jan26 recording (209 events) and the
matching Raster_EvtNNN image for each event.

Run:
    python pca_explorer_gui.py
    python pca_explorer_gui.py --csv other_points.csv --images path/to/images

Controls:
    • Click a dot            → show that event's image
    • Left / Right arrows    → step to previous / next event (by chronological #)
    • "Color by" dropdown    → recolor dots by final type / KMeans / DBSCAN

Author: Shahriar Tafti, UVM
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

HERE = Path(__file__).resolve().parent

# DS-type color scheme (Toothy uses a blue/red 'bwr' split): 1 = blue, 2 = red,
# 0 = gray (DBSCAN noise / unclassified).
TYPE_COLORS = {0: "#9e9e9e", 1: "#1f77d4", 2: "#d62728"}
TYPE_LABELS = {0: "Unclassified", 1: "DS Type 1", 2: "DS Type 2"}


class PCAExplorer(QtWidgets.QMainWindow):
    def __init__(self, csv_path: Path, image_dir: Path):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.df = self._load_points(csv_path)
        self.color_col = "type"
        self.sel = None                     # currently selected row index (0-based)
        self._pixmap = None                 # current QPixmap (kept for rescaling)

        self.setWindowTitle("PCA Explorer — Dentate Spike Classification (M2s8jan26)")
        self.resize(1280, 760)
        self._build_ui()
        self._draw_scatter()
        if len(self.df):
            self._select(0)

    # ── data ──────────────────────────────────────────────────────────────────
    def _load_points(self, csv_path):
        df = pd.read_csv(csv_path)
        for c in ("type", "k_type", "db_type"):
            if c not in df.columns:
                df[c] = df.get("type", 1)
        return df.reset_index(drop=True)

    # ── UI layout ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ---- LEFT: scatter plot -------------------------------------------------
        left = QtWidgets.QVBoxLayout()

        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Color by:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(["Final type", "KMeans (k_type)", "DBSCAN (db_type)"])
        self.color_combo.currentIndexChanged.connect(self._on_color_changed)
        ctl.addWidget(self.color_combo)
        ctl.addStretch(1)
        self.count_lbl = QtWidgets.QLabel()
        ctl.addWidget(self.count_lbl)
        left.addLayout(ctl)

        self.fig = Figure(figsize=(6, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        left.addWidget(NavigationToolbar(self.canvas, self))
        left.addWidget(self.canvas, 1)

        root.addLayout(left, 3)

        # ---- RIGHT: image + info ------------------------------------------------
        right = QtWidgets.QVBoxLayout()
        self.info_lbl = QtWidgets.QLabel("Click a dot to view its event image.")
        self.info_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.info_lbl.setWordWrap(True)
        right.addWidget(self.info_lbl)

        self.img_lbl = QtWidgets.QLabel()
        self.img_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.img_lbl.setMinimumSize(360, 360)
        self.img_lbl.setStyleSheet(
            "background:#111; color:#aaa; border:1px solid #444;")
        self.img_lbl.setText("(no image)")
        right.addWidget(self.img_lbl, 1)

        nav = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        right.addLayout(nav)

        root.addLayout(right, 2)

    # ── scatter drawing ──────────────────────────────────────────────────────────
    def _colors(self):
        vals = self.df[self.color_col].astype(int).values
        return [TYPE_COLORS.get(v, "#9e9e9e") for v in vals], vals

    def _draw_scatter(self):
        self.ax.clear()
        colors, vals = self._colors()
        self.ax.scatter(self.df["pc1"], self.df["pc2"], c=colors,
                        s=42, alpha=0.85, edgecolors="white", linewidths=0.4,
                        picker=False, zorder=2)
        # legend
        present = sorted(set(vals))
        handles = [matplotlib.lines.Line2D([], [], marker="o", linestyle="",
                   markerfacecolor=TYPE_COLORS.get(v, "#9e9e9e"),
                   markeredgecolor="white", markersize=8,
                   label=f"{TYPE_LABELS.get(v, v)}  (n={int((vals==v).sum())})")
                   for v in present]
        self.ax.legend(handles=handles, loc="upper right", fontsize=9,
                       framealpha=0.9)
        self.ax.set_xlabel("Principal Component 1")
        self.ax.set_ylabel("Principal Component 2")
        self.ax.set_title("PCA of Dentate-Spike CSD profiles\n(click a point)",
                          fontweight="bold")
        self.ax.grid(True, alpha=0.25)

        # selection marker (ring) drawn on top; created empty, updated on select
        (self._sel_marker,) = self.ax.plot([], [], "o", markersize=16,
                                           markerfacecolor="none",
                                           markeredgecolor="#111", markeredgewidth=2,
                                           zorder=3)
        self.count_lbl.setText(f"{len(self.df)} events")
        if self.sel is not None:
            self._update_marker()
        self.canvas.draw_idle()

    def _update_marker(self):
        if self.sel is None:
            return
        row = self.df.iloc[self.sel]
        self._sel_marker.set_data([row["pc1"]], [row["pc2"]])
        self.canvas.draw_idle()

    # ── interaction ──────────────────────────────────────────────────────────────
    def _on_color_changed(self, i):
        self.color_col = ["type", "k_type", "db_type"][i]
        self._draw_scatter()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        # nearest point in display (pixel) space so aspect ratio doesn't bias it
        xy = self.ax.transData.transform(self.df[["pc1", "pc2"]].values)
        click = np.array([event.x, event.y])
        i = int(np.argmin(np.hypot(xy[:, 0] - click[0], xy[:, 1] - click[1])))
        # ignore clicks far from any point
        if np.hypot(*(xy[i] - click)) > 40:
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
        self.sel = i
        row = self.df.iloc[i]
        self._update_marker()
        self.info_lbl.setText(
            f"Event #{int(row['event'])} / {len(self.df)}    "
            f"[{TYPE_LABELS.get(int(row['type']), row['type'])}]\n"
            f"t = {row['time_sec']:.3f} s   (sample idx {int(row['idx'])})\n"
            f"PC1 = {row['pc1']:.3f}   PC2 = {row['pc2']:.3f}\n"
            f"KMeans={int(row['k_type'])}  DBSCAN={int(row['db_type'])}")
        self._show_image(row["image"])

    # ── image panel ──────────────────────────────────────────────────────────────
    def _show_image(self, filename):
        path = self.image_dir / str(filename)
        if not path.exists():
            self._pixmap = None
            self.img_lbl.setText(f"(image not found)\n{filename}")
            return
        pm = QtGui.QPixmap(str(path))
        if pm.isNull():
            self._pixmap = None
            self.img_lbl.setText(f"(could not load)\n{filename}")
            return
        self._pixmap = pm
        self._rescale_image()

    def _rescale_image(self):
        if self._pixmap is None:
            return
        self.img_lbl.setPixmap(self._pixmap.scaled(
            self.img_lbl.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_image()


def main():
    ap = argparse.ArgumentParser(description="Interactive PCA → event-image explorer")
    ap.add_argument("--csv", default=str(HERE / "pca_points.csv"),
                    help="points CSV from prepare_pca_points.py")
    ap.add_argument("--images", default=str(HERE / "Test Data" / "M2s8jan26"),
                    help="folder containing Raster_EvtNNN images")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run prepare_pca_points.py first.")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    win = PCAExplorer(csv_path, Path(args.images))
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
