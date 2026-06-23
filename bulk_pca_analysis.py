"""
Toothy (https://github.com/Farrell-Laboratory/Toothy)

Purpose: Bulk PCA analysis — batch CSD + PCA classification that compares
         manual channel inputs (from the feeder-sheet CSV) against Toothy
         auto-estimated channels, and exports full spike timestamps.

         Outputs per run:
           bulk_pca_results.h5   — timestamps + sample indices for every
                                   session × {manual, auto} × {DS1, DS2}
           bulk_pca_summary.csv  — one row per session with counts, channels,
                                   and comparison metadata

Authors: Shahriar Tafti, UVM

Last updated: 2025-06-01
"""

import sys
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import quantities as pq
from PyQt5 import QtWidgets, QtCore, QtGui

import QSS
import pyfx
import ephys
import gui_items as gi
import data_processing as dp

# Fixed seed — guarantees identical KMeans results across runs for the same data
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-channel estimation
# ─────────────────────────────────────────────────────────────────────────────

def _load_auto_channels(ddir, iprb=0):
    """
    Estimate fissure / ripple / hilus channels the way the channel selection
    GUI would on first open.  Returns (fissure_ch, ripple_ch, hil_ch) as
    0-indexed absolute channel numbers.  Any channel that cannot be estimated
    is returned as None.
    """
    hdf5_path = Path(ddir, 'DATA.hdf5')
    if not hdf5_path.exists():
        return None, None, None

    fissure_ch = ripple_ch = hil_ch = None
    try:
        STD = pd.read_hdf(str(hdf5_path), key=ephys.get_h5_key('STD', iprb=iprb))
        fissure_ch = int(ephys.estimate_theta_chan(STD))
        ripple_ch  = int(ephys.estimate_ripple_chan(STD))
    except Exception:
        pass
    try:
        DF_ALL = pd.read_hdf(str(hdf5_path), key=ephys.get_h5_key('ALL_DS', iprb=iprb))
        STD_for_hil = pd.read_hdf(str(hdf5_path), key=ephys.get_h5_key('STD', iprb=iprb))
        probe = ephys.read_probe_group(ddir).probes[iprb]
        DF_ALL = ephys.clean_event_df(DF_ALL, STD_for_hil, probe)
        DS_MEAN = ephys.get_mean_event_df(DF_ALL, STD_for_hil)
        hil_ch = int(ephys.estimate_hil_chan(DS_MEAN))
    except Exception:
        pass
    return fissure_ch, ripple_ch, hil_ch


# ─────────────────────────────────────────────────────────────────────────────
#  Core analysis function
# ─────────────────────────────────────────────────────────────────────────────

def _run_pca(ddir, fissure_chan, ripple_chan, hil_chan,
             bad_channels=None, iprb=0, ishank=0, progress_fn=None):
    """
    Run CSD + PCA classification for one recording with explicit channel
    assignments.  DS events are re-derived from ALL_DS filtered to hil_chan
    so this works for any channel combination, independent of any pre-saved
    DS_DF file.

    Parameters
    ----------
    ddir          : toothy sub-directory (contains DATA.hdf5)
    fissure_chan  : 0-indexed absolute channel number for the fissure
    ripple_chan   : 0-indexed absolute channel number for the ripple (unused
                   in DS classification, stored for metadata only)
    hil_chan      : 0-indexed absolute channel number for the hilus
    bad_channels  : list of 0-indexed absolute channel numbers to mark noisy
                   (applied in memory only, not written to disk)
    random_state  : fixed via module-level RANDOM_STATE

    Returns
    -------
    dict with keys:
        ds1_n, ds2_n,
        ds1_idx, ds2_idx          (int arrays of sample indices),
        ds1_timestamps_sec,
        ds2_timestamps_sec        (float arrays, seconds),
        lfp_fs
    Raises ValueError on unrecoverable problems.
    """

    def _step(msg):
        if progress_fn:
            progress_fn(msg)

    # ── probe / shank geometry ───────────────────────────────────────────────
    probe          = ephys.read_probe_group(ddir).probes[iprb]
    shank          = probe.get_shanks()[ishank]
    ypos           = np.array(sorted(shank.contact_positions[:, 1]))
    coord_elec     = pq.Quantity(ypos, probe.si_units).rescale('m', dtype='float32')
    shank_channels = shank.get_indices()
    channels       = np.arange(len(shank_channels), dtype='int')

    if fissure_chan not in list(shank_channels):
        raise ValueError(f'fissure_chan={fissure_chan} not in shank channels')
    if hil_chan not in list(shank_channels):
        raise ValueError(f'hil_chan={hil_chan} not in shank channels')

    rel_theta_chan = list(shank_channels).index(fissure_chan)

    # ── load LFP + noise ─────────────────────────────────────────────────────
    dmode = dp.validate_processed_ddir(ddir)
    if dmode not in (1, 2):
        raise ValueError('No valid processed data found in directory')

    _step('  Loading LFP...')
    hdf5_path = Path(ddir, 'DATA.hdf5')

    if dmode == 1:
        with h5py.File(str(hdf5_path), 'r') as FF:
            lfp_fs      = int(FF.attrs['lfp_fs'])
            lfp_all     = ephys.load_h5_lfp(FF, key='raw', iprb=iprb)[shank_channels, :]
            NOISE_ALL   = ephys.load_h5_array(FF, 'NOISE', iprb, in_memory=True)
            NOISE_TRAIN = NOISE_ALL[shank_channels].copy()
    else:
        lfp_all     = ephys.load_bp(ddir, key='raw', iprb=iprb)[shank_channels, :]
        lfp_fs      = int(np.load(Path(ddir, 'lfp_fs.npy')))
        NOISE_ALL   = ephys.load_noise_channels(ddir, iprb=iprb).copy()
        NOISE_TRAIN = NOISE_ALL[shank_channels].copy()

    _step(f'  LFP: {lfp_all.shape[0]} ch × {lfp_all.shape[1]:,} samples  '
          f'({lfp_fs} Hz, {lfp_all.shape[1]/lfp_fs/60:.1f} min)')

    # ── apply bad channels (in memory only — nothing written to disk) ────────
    if bad_channels:
        newly = []
        for ch in bad_channels:
            if ch in shank_channels and not NOISE_ALL[ch]:
                NOISE_ALL[ch] = 1
                NOISE_TRAIN[list(shank_channels).index(ch)] = 1
                newly.append(ch)
        if newly:
            _step(f'  Marked noisy (analysis only): {newly}')

    # ── noise interpolation ──────────────────────────────────────────────────
    noise_idx = np.nonzero(NOISE_TRAIN)[0]
    clean_idx = np.setdiff1d(channels, noise_idx)
    if len(noise_idx):
        _step(f'  Interpolating {len(noise_idx)} noisy channel(s)')
    lfp_interp = deepcopy(lfp_all)
    for i in noise_idx:
        if len(clean_idx) == 0:
            lfp_interp[i] = np.zeros_like(lfp_interp[i])
        elif i < int(min(clean_idx)):
            lfp_interp[i] = lfp_all[int(min(clean_idx))]
        elif i > int(max(clean_idx)):
            lfp_interp[i] = lfp_all[int(max(clean_idx))]
        else:
            s1 = lfp_all[int(pyfx.Closest(i, clean_idx[clean_idx < i]))]
            s2 = lfp_all[int(pyfx.Closest(i, clean_idx[clean_idx > i]))]
            lfp_interp[i] = np.nanmean([s1, s2], axis=0)

    # ── build DS_DF from ALL_DS ──────────────────────────────────────────────
    if not hdf5_path.exists():
        raise ValueError('DATA.hdf5 not found — only HDF5 format is supported')

    DF_ALL = pd.read_hdf(str(hdf5_path), key=ephys.get_h5_key('ALL_DS', iprb=iprb))
    STD    = pd.read_hdf(str(hdf5_path), key=ephys.get_h5_key('STD',    iprb=iprb))
    DS_ALL = ephys.clean_event_df(DF_ALL, STD, probe)

    if hil_chan in DS_ALL.index:
        DS_DF = DS_ALL.loc[[hil_chan]].copy()
        DS_DF = DS_DF[DS_DF['is_valid'] == 1].reset_index(drop=True)
    else:
        DS_DF = pd.DataFrame(columns=DS_ALL.columns)

    if 'idx' in DS_DF.columns:
        DS_DF['idx'] = DS_DF['idx'].astype(int)

    if len(DS_DF) < 2:
        raise ValueError(
            f'Too few valid DS events ({len(DS_DF)}) at hil_chan={hil_chan + 1}'
        )

    iev   = np.atleast_1d(DS_DF.idx.values)
    ddict = dict(ephys.load_recording_params(ddir))
    _step(f'  DS events: {len(iev):,}  '
          f'(fissure={fissure_chan+1}, ripple='
          f'{ripple_chan+1 if ripple_chan is not None else "N/A"}, '
          f'hilus={hil_chan+1})')

    # ── CSD ──────────────────────────────────────────────────────────────────
    csd_chs = np.arange(rel_theta_chan, len(channels))
    if len(csd_chs) < 2:
        raise ValueError(
            f'CSD window has only {len(csd_chs)} channel(s) '
            f'(fissure at index {rel_theta_chan}, total {len(channels)} channels)'
        )

    def _csd(idx):
        data = lfp_interp[csd_chs, :][:, np.atleast_1d(idx)]
        obj  = ephys.get_csd_obj(data, coord_elec[csd_chs], ddict)
        return ephys.csd_obj2arrs(obj)

    _step(f'  Computing CSD ({len(csd_chs)} channels × {len(iev):,} events)...')
    raw_csd, filt_csd, norm_filt_csd = _csd(iev)

    # ── PCA ──────────────────────────────────────────────────────────────────
    norm_csd = norm_filt_csd if norm_filt_csd.ndim == 2 else norm_filt_csd[:, np.newaxis]
    n_events = norm_csd.shape[1]
    if min(norm_csd.T.shape) < 2:
        raise ValueError(f'Too few events for PCA (n_events={n_events})')

    _step(f'  PCA + KMeans (seed={RANDOM_STATE})...')
    pca_fit = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(norm_csd.T)
    algo    = str(ddict.get('clus_algo', 'kmeans'))

    def _set_ds_type(types):
        r1, r2 = np.where(types == 1)[0], np.where(types == 2)[0]
        if not len(r1) or not len(r2):
            return types
        c1 = _csd(DS_DF.idx.values[r1])[2]
        c2 = _csd(DS_DF.idx.values[r2])[2]
        if np.argmin(np.nanmean(c1, axis=1)) > np.argmin(np.nanmean(c2, axis=1)):
            types[r1] = 2
            types[r2] = 1
        return types

    km = KMeans(
        n_clusters=int(ddict.get('nclusters', 2)),
        n_init='auto',
        random_state=RANDOM_STATE,
    ).fit(pca_fit)
    db = DBSCAN(
        eps=float(ddict.get('eps', 0.2)),
        min_samples=int(ddict.get('min_clus_samples', 3)),
    ).fit(pca_fit)

    km_types = _set_ds_type(np.array([{0: 2, 1: 1}.get(x, 0) for x in km.labels_]))
    db_types = _set_ds_type(np.array([{0: 1, 1: 2}.get(x, 0) for x in db.labels_]))
    dstypes  = km_types if algo == 'kmeans' else db_types
    DS_DF['type'] = dstypes

    # ── collect timestamps ───────────────────────────────────────────────────
    mask1 = DS_DF['type'] == 1
    mask2 = DS_DF['type'] == 2
    ds1_idx = DS_DF.idx.values[mask1].astype(int)
    ds2_idx = DS_DF.idx.values[mask2].astype(int)

    _step(f'  DS1={int(mask1.sum())}, DS2={int(mask2.sum())}')

    return {
        'ds1_n'              : int(mask1.sum()),
        'ds2_n'              : int(mask2.sum()),
        'ds1_idx'            : ds1_idx,
        'ds2_idx'            : ds2_idx,
        'ds1_timestamps_sec' : ds1_idx.astype(float) / lfp_fs,
        'ds2_timestamps_sec' : ds2_idx.astype(float) / lfp_fs,
        'lfp_fs'             : lfp_fs,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HDF5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_run_to_h5(grp, key, res, fissure_ch, ripple_ch, hil_ch):
    """Write one run's results into an HDF5 group under `key` ('manual'/'auto')."""
    if key in grp:
        del grp[key]
    g = grp.create_group(key)
    # store as 1-indexed display values to match GUI
    g.attrs['fissure_ch'] = int(fissure_ch) + 1
    g.attrs['ripple_ch']  = int(ripple_ch)  + 1 if ripple_ch  is not None else -1
    g.attrs['hil_ch']     = int(hil_ch)     + 1 if hil_ch     is not None else -1
    g.attrs['ds1_n']      = res['ds1_n']
    g.attrs['ds2_n']      = res['ds2_n']
    g.attrs['lfp_fs']     = res['lfp_fs']
    g.create_dataset('DS1_sample_idx',      data=res['ds1_idx'],            compression='gzip')
    g.create_dataset('DS2_sample_idx',      data=res['ds2_idx'],            compression='gzip')
    g.create_dataset('DS1_timestamps_sec',  data=res['ds1_timestamps_sec'], compression='gzip')
    g.create_dataset('DS2_timestamps_sec',  data=res['ds2_timestamps_sec'], compression='gzip')


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

class BulkPCAPopup(QtWidgets.QDialog):
    """
    Batch CSD + PCA classification.  For each recording in the feeder-sheet:
      1. Runs classification with the manual channels provided in the CSV.
      2. Runs classification with Toothy auto-estimated channels.
      3. Saves full DS1/DS2 spike timestamps to an HDF5 file and writes a
         summary CSV comparing the two channel sets.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Bulk PCA Analysis')
        self._df = None
        self._build_ui()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(12)
        outer.setContentsMargins(16, 16, 16, 16)

        info_lbl = QtWidgets.QLabel(
            'Load a feeder-sheet CSV, choose an output directory, then click Run.\n'
            'Each recording is classified twice — with manual channel inputs and\n'
            'with Toothy auto-estimated channels — so you can compare.\n'
            f'Clustering uses a fixed random seed ({RANDOM_STATE}) for reproducibility.'
        )
        info_lbl.setWordWrap(True)
        outer.addWidget(info_lbl)

        for attr, lbl_text, placeholder, browse_slot in [
            ('csv_le',  'Input CSV:',        'Browse for feeder-sheet CSV...',       self._browse_csv),
            ('out_le',  'Output folder:',    'Choose where to save results...',      self._browse_output),
        ]:
            frame = QtWidgets.QFrame()
            hl    = QtWidgets.QHBoxLayout(frame)
            hl.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setFixedWidth(100)
            le  = QtWidgets.QLineEdit()
            le.setReadOnly(True)
            le.setPlaceholderText(placeholder)
            btn = QtWidgets.QPushButton('Browse')
            btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
            btn.setFixedWidth(70)
            btn.clicked.connect(browse_slot)
            hl.addWidget(lbl)
            hl.addWidget(le, stretch=1)
            hl.addWidget(btn)
            outer.addWidget(frame)
            setattr(self, attr, le)

        self.status_txt = QtWidgets.QTextEdit()
        self.status_txt.setReadOnly(True)
        self.status_txt.setPlaceholderText(
            'Load a CSV and choose an output folder, then click Run...'
        )
        self.status_txt.setMinimumHeight(300)
        outer.addWidget(self.status_txt, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        self.close_btn = QtWidgets.QPushButton('Close')
        self.close_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        self.close_btn.clicked.connect(self.reject)
        self.run_btn = QtWidgets.QPushButton('Run')
        self.run_btn.setStyleSheet(
            pyfx.dict2ss(QSS.TOGGLE_BTN) + 'QPushButton { background-color: #c8f0c8; }'
        )
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        btn_row.addWidget(self.close_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        outer.addLayout(btn_row)

        self.setMinimumSize(660, 560)

    # ── file browsing ────────────────────────────────────────────────────────

    def _browse_csv(self):
        fpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select feeder-sheet CSV', os.path.expanduser('~'), 'CSV files (*.csv)'
        )
        if not fpath:
            return
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            gi.MsgboxError(f'Could not read CSV:\n{e}', parent=self).exec()
            return
        if 'Path' not in df.columns:
            gi.MsgboxError("CSV must have a 'Path' column.", parent=self).exec()
            return
        self._df = df.dropna(subset=['Path']).reset_index(drop=True)
        self.csv_le.setText(fpath)
        self.status_txt.setText(f'Loaded {len(self._df)} rows.')
        self._update_run_btn()

    def _browse_output(self):
        dpath = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select output folder', os.path.expanduser('~')
        )
        if dpath:
            self.out_le.setText(dpath)
        self._update_run_btn()

    def _update_run_btn(self):
        self.run_btn.setEnabled(
            self._df is not None and len(self._df) > 0
            and bool(self.out_le.text().strip())
        )

    # ── live logging ─────────────────────────────────────────────────────────

    def _log(self, text):
        self.status_txt.append(text)
        QtWidgets.QApplication.processEvents()

    # ── main run ─────────────────────────────────────────────────────────────

    def _run(self):
        if self._df is None or not self.out_le.text().strip():
            return

        out_dir  = self.out_le.text().strip()
        h5_path  = str(Path(out_dir, 'bulk_pca_results.h5'))
        csv_path = str(Path(out_dir, 'bulk_pca_summary.csv'))

        self.status_txt.clear()
        self.run_btn.setEnabled(False)

        summary_rows = []
        n_total = len(self._df)
        n_ok = n_skip = n_err = 0

        with h5py.File(h5_path, 'w') as H:
            H.attrs['random_state'] = RANDOM_STATE
            H.attrs['created_by']   = 'Toothy Bulk PCA Analysis'
            H.attrs['author']       = 'Shahriar Tafti, UVM'

            for i, (_, row) in enumerate(self._df.iterrows()):
                raw_path    = str(row['Path']).strip()
                sid_val     = row.get('Session ID', None)
                session_id  = str(sid_val) if pd.notna(sid_val) else os.path.basename(raw_path)
                toothy_ddir = str(Path(raw_path, 'toothy'))
                notes       = str(row.get('Notes', '')) if pd.notna(row.get('Notes', pd.NA)) else ''

                self._log(f'\n[{i+1}/{n_total}] {session_id}\n  {raw_path}')

                # ── bad channels ─────────────────────────────────────────────
                bad_channels = []
                bc_val = row.get('Bad Channels', None)
                if pd.notna(bc_val) and str(bc_val).strip():
                    try:
                        bad_channels = [int(x.strip()) for x in str(bc_val).split(',')]
                    except ValueError:
                        pass
                if bad_channels:
                    self._log(f'  Bad channels: {bad_channels}')

                # ── parse manual channels (CSV uses 1-indexed display values) ─
                def _parse_ch(col):
                    v = row.get(col, None)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return None
                    try:
                        return int(float(str(v).strip())) - 1
                    except (ValueError, TypeError):
                        return None

                man_fissure = _parse_ch('Fissure Channel')
                man_ripple  = _parse_ch('Ripple Channel')
                man_hilus   = _parse_ch('Hilus Channel')

                row_data = {
                    'Session ID'    : session_id,
                    'Path'          : raw_path,
                    'Notes'         : notes,
                    'Bad Channels'  : str(bad_channels) if bad_channels else '',
                    'Manual Fissure': (man_fissure + 1) if man_fissure is not None else '',
                    'Manual Ripple' : (man_ripple  + 1) if man_ripple  is not None else '',
                    'Manual Hilus'  : (man_hilus   + 1) if man_hilus   is not None else '',
                    'Manual DS1 N'  : '', 'Manual DS2 N': '', 'Manual Total N': '',
                    'Auto Fissure'  : '', 'Auto Ripple' : '', 'Auto Hilus'    : '',
                    'Auto DS1 N'    : '', 'Auto DS2 N'  : '', 'Auto Total N'  : '',
                    'Channels Match': '', 'Status': '', 'Error': '',
                }

                if man_fissure is None or man_hilus is None:
                    self._log('[SKIP] Missing Fissure or Hilus channel in CSV')
                    row_data.update(Status='SKIP', Error='Missing Fissure or Hilus in CSV')
                    summary_rows.append(row_data)
                    n_skip += 1
                    continue

                # ── auto-estimated channels ──────────────────────────────────
                self._log('  Estimating auto channels...')
                auto_fissure, auto_ripple, auto_hilus = _load_auto_channels(toothy_ddir)
                if auto_fissure is not None:
                    self._log(
                        f'  Auto  — fissure={auto_fissure+1}, '
                        f'ripple={auto_ripple+1 if auto_ripple is not None else "N/A"}, '
                        f'hilus={auto_hilus+1 if auto_hilus is not None else "N/A"}'
                    )
                    row_data.update(
                        **{'Auto Fissure': auto_fissure + 1,
                           'Auto Ripple' : (auto_ripple + 1) if auto_ripple is not None else '',
                           'Auto Hilus'  : (auto_hilus  + 1) if auto_hilus  is not None else ''}
                    )
                else:
                    self._log('  Auto-estimation unavailable')

                # ── HDF5 session group ───────────────────────────────────────
                grp = H.require_group(session_id)
                grp.attrs['path']         = raw_path
                grp.attrs['notes']        = notes
                grp.attrs['bad_channels'] = str(bad_channels)

                # ── manual run ───────────────────────────────────────────────
                man_res   = None
                man_error = ''
                self._log(f'  Manual — fissure={man_fissure+1}, '
                          f'ripple={man_ripple+1 if man_ripple is not None else "N/A"}, '
                          f'hilus={man_hilus+1}')
                try:
                    man_res = _run_pca(
                        toothy_ddir,
                        fissure_chan=man_fissure,
                        ripple_chan=man_ripple,
                        hil_chan=man_hilus,
                        bad_channels=bad_channels or None,
                        progress_fn=self._log,
                    )
                    _save_run_to_h5(grp, 'manual', man_res,
                                    man_fissure, man_ripple, man_hilus)
                    row_data.update(**{
                        'Manual DS1 N'  : man_res['ds1_n'],
                        'Manual DS2 N'  : man_res['ds2_n'],
                        'Manual Total N': man_res['ds1_n'] + man_res['ds2_n'],
                    })
                except Exception as e:
                    man_error = str(e)
                    self._log(f'  [Manual] ERROR: {e}')

                # ── auto run ────────────────────────────────────────────────
                auto_res   = None
                auto_error = ''
                if auto_fissure is not None and auto_hilus is not None:
                    same = (auto_fissure == man_fissure
                            and auto_hilus   == man_hilus
                            and auto_ripple  == man_ripple)
                    row_data['Channels Match'] = 'Yes' if same else 'No'
                    if same and man_res is not None:
                        self._log('  Auto channels match manual — reusing results')
                        auto_res = man_res
                        _save_run_to_h5(grp, 'auto', man_res,
                                        auto_fissure, auto_ripple, auto_hilus)
                        row_data.update(**{
                            'Auto DS1 N'  : man_res['ds1_n'],
                            'Auto DS2 N'  : man_res['ds2_n'],
                            'Auto Total N': man_res['ds1_n'] + man_res['ds2_n'],
                        })
                    elif not same:
                        try:
                            auto_res = _run_pca(
                                toothy_ddir,
                                fissure_chan=auto_fissure,
                                ripple_chan=auto_ripple,
                                hil_chan=auto_hilus,
                                bad_channels=bad_channels or None,
                                progress_fn=self._log,
                            )
                            _save_run_to_h5(grp, 'auto', auto_res,
                                            auto_fissure, auto_ripple, auto_hilus)
                            row_data.update(**{
                                'Auto DS1 N'  : auto_res['ds1_n'],
                                'Auto DS2 N'  : auto_res['ds2_n'],
                                'Auto Total N': auto_res['ds1_n'] + auto_res['ds2_n'],
                            })
                        except Exception as e:
                            auto_error = str(e)
                            self._log(f'  [Auto] ERROR: {e}')
                            row_data.update(**{
                                'Auto DS1 N': 'ERROR',
                                'Auto DS2 N': 'ERROR',
                                'Auto Total N': 'ERROR',
                            })
                else:
                    row_data['Channels Match'] = 'N/A'
                    row_data.update(**{
                        'Auto DS1 N': 'N/A', 'Auto DS2 N': 'N/A', 'Auto Total N': 'N/A'
                    })

                # ── finalise row ────────────────────────────────────────────
                combined_err = ' | '.join(filter(None, [man_error, auto_error]))
                if man_res is not None:
                    row_data['Status'] = 'OK'
                    n_ok += 1
                elif man_error:
                    row_data['Status'] = 'ERROR'
                    row_data['Error']  = combined_err
                    n_err += 1

                summary_rows.append(row_data)

        self._log(f'\nDone: {n_ok} OK, {n_skip} skipped, {n_err} errors')

        # ── write CSV summary ────────────────────────────────────────────────
        col_order = [
            'Session ID', 'Path', 'Notes', 'Bad Channels',
            'Manual Fissure', 'Manual Ripple', 'Manual Hilus',
            'Manual DS1 N', 'Manual DS2 N', 'Manual Total N',
            'Auto Fissure', 'Auto Ripple', 'Auto Hilus',
            'Auto DS1 N', 'Auto DS2 N', 'Auto Total N',
            'Channels Match', 'Status', 'Error',
        ]
        summary_df = pd.DataFrame(summary_rows)
        present    = [c for c in col_order if c in summary_df.columns]
        summary_df[present].to_csv(csv_path, index=False)

        self._log(f'\nResults HDF5 → {h5_path}')
        self._log(f'Summary CSV  → {csv_path}')


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = pyfx.qapp()
    w = BulkPCAPopup()
    w.show()
    w.raise_()
    sys.exit(app.exec())
