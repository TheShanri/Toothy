#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toothy (https://github.com/Farrell-Laboratory/Toothy)

PCA Module — reconstructed core of Toothy's DS (Dentate Spike) classification.

This distills the PCA + clustering logic that is otherwise duplicated inside:
    * ds_classification_gui.py  ->  CSDPlotWidget.run_pca()   (interactive GUI)
    * bulk_pca_analysis.py      ->  _run_pca()                (headless batch)

The pipeline is:
    normalized CSD  ->  PCA (2 components)  ->  KMeans + DBSCAN clustering
    ->  DS Type 1 / Type 2 labels (oriented by CSD sink depth).

Nothing here depends on Qt, ephys, or the file system, so it can be unit-tested
and reused directly. The only project-specific step (orienting the two clusters
by sink depth) is injected as an optional callback.

Author: Shahriar Tafti, UVM
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


# Default clustering parameters (mirror qparam / CSD_PARAMS defaults).
DEFAULT_PARAMS = {
    'nclusters'        : 2,        # KMeans number of clusters
    'eps'              : 0.2,      # DBSCAN neighborhood radius
    'min_clus_samples' : 3,        # DBSCAN minimum core-point samples
    'clus_algo'        : 'kmeans', # which labeling becomes the final 'type'
}


def run_pca(norm_csd, params=None, random_state=None, orient_fn=None):
    """
    Run PCA + clustering on a normalized CSD matrix to classify DS events.

    Parameters
    ----------
    norm_csd : ndarray, shape (n_channels, n_events)
        The normalized, filtered CSD (column-normalized per event). This is the
        `norm_filt_csd` returned by ephys.csd_obj2arrs(). A 1-D array (single
        event) is promoted to a column vector.
    params : dict, optional
        Clustering parameters. Recognized keys (see DEFAULT_PARAMS):
        'nclusters', 'eps', 'min_clus_samples', 'clus_algo'.
    random_state : int or None, optional
        Seed for PCA and KMeans. None -> non-deterministic (GUI behavior);
        an int -> reproducible (bulk_pca_analysis uses RANDOM_STATE).
    orient_fn : callable(types, idx_rows_type1, idx_rows_type2) -> bool, optional
        Callback that returns True if the two clusters' Type-1/Type-2 labels
        should be swapped, based on CSD sink depth. In Toothy the deeper sink
        (larger channel index) defines DS Type 1. If None, no reorientation is
        performed (labels come straight from the clusterer).

    Returns
    -------
    result : dict
        'pc1'     : ndarray (n_events,)  first principal component per event
        'pc2'     : ndarray (n_events,)  second principal component per event
        'k_type'  : ndarray (n_events,)  KMeans labels mapped to {1, 2}
        'db_type' : ndarray (n_events,)  DBSCAN labels mapped to {0, 1, 2}
                                         (0 = noise / unclustered)
        'type'    : ndarray (n_events,)  final labels (k_type or db_type,
                                         chosen by params['clus_algo'])
        'pca'     : the fitted sklearn PCA object (components_, explained_
                                         variance_ratio_, etc.)
        'kmeans'  : the fitted KMeans object
        'dbscan'  : the fitted DBSCAN object

    Raises
    ------
    ValueError
        If there are fewer than 2 events (PCA needs >= 2 samples).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    # Ensure 2D (n_channels, n_events); a single event collapses to 1D.
    norm_csd = np.asarray(norm_csd)
    if norm_csd.ndim == 1:
        norm_csd = norm_csd[:, np.newaxis]

    # PCA operates on (n_events, n_channels): rows = samples, cols = features.
    X = norm_csd.T
    n_events, n_features = X.shape
    if min(n_events, n_features) < 2:
        raise ValueError(f'Too few events for PCA (n_events={n_events}).')

    # ── Principal component analysis (2 components) ──────────────────────────
    pca = PCA(n_components=2, random_state=random_state)
    pca_fit = pca.fit_transform(X)          # shape (n_events, 2)

    # ── Unsupervised clustering on the 2 PCs ─────────────────────────────────
    kmeans = KMeans(
        n_clusters=int(p['nclusters']),
        n_init='auto',
        random_state=random_state,
    ).fit(pca_fit)
    dbscan = DBSCAN(
        eps=float(p['eps']),
        min_samples=int(p['min_clus_samples']),
    ).fit(pca_fit)

    # Map raw cluster ids to DS types. KMeans: {0->2, 1->1}; DBSCAN:
    # {0->1, 1->2}; any other id (incl. DBSCAN noise -1) -> 0.
    k_types  = np.array([{0: 2, 1: 1}.get(x, 0) for x in kmeans.labels_])
    db_types = np.array([{0: 1, 1: 2}.get(x, 0) for x in dbscan.labels_])

    # Orient labels so DS Type 1 corresponds to the deeper sink, if a callback
    # is supplied (this is the get_csd()/sink-depth step in the Toothy code).
    if orient_fn is not None:
        k_types  = _apply_orientation(k_types, orient_fn)
        db_types = _apply_orientation(db_types, orient_fn)

    # Final classification follows the selected algorithm.
    dstypes = k_types if p['clus_algo'] == 'kmeans' else db_types

    return {
        'pc1'     : pca_fit[:, 0],
        'pc2'     : pca_fit[:, 1],
        'k_type'  : k_types,
        'db_type' : db_types,
        'type'    : dstypes,
        'pca'     : pca,
        'kmeans'  : kmeans,
        'dbscan'  : dbscan,
    }


def _apply_orientation(types, orient_fn):
    """Swap Type-1 / Type-2 rows if orient_fn says the sink order is reversed."""
    rows1 = np.where(types == 1)[0]
    rows2 = np.where(types == 2)[0]
    if len(rows1) == 0 or len(rows2) == 0:
        return types
    if orient_fn(types, rows1, rows2):
        types = types.copy()
        types[rows1] = 2
        types[rows2] = 1
    return types


def make_sink_orient_fn(get_csd_fn):
    """
    Build an orient_fn from a CSD-lookup callable, replicating set_ds_type().

    Parameters
    ----------
    get_csd_fn : callable(rows) -> ndarray (n_channels, n_selected_events)
        Returns the CSD for the events at the given row indices (in Toothy this
        wraps self.get_csd(...)[2] / _csd(...)[2]).

    Returns
    -------
    orient_fn : callable suitable for run_pca(orient_fn=...).
        Returns True when the Type-1 sink lies *below* the Type-2 sink
        (larger channel index == lower/deeper sink), signaling a label swap.
    """
    def orient_fn(types, rows1, rows2):
        csd1 = get_csd_fn(rows1)
        csd2 = get_csd_fn(rows2)
        imin1 = np.argmin(np.nanmean(csd1, axis=1))
        imin2 = np.argmin(np.nanmean(csd2, axis=1))
        return imin1 > imin2
    return orient_fn


if __name__ == '__main__':
    # Tiny smoke test with synthetic two-blob CSD data.
    rng = np.random.default_rng(0)
    n_ch = 16
    blob1 = rng.normal(0.0, 0.1, size=(n_ch, 40))
    blob2 = rng.normal(1.0, 0.1, size=(n_ch, 40))
    norm_csd = np.hstack([blob1, blob2])

    res = run_pca(norm_csd, random_state=42)
    print('explained variance ratio:', res['pca'].explained_variance_ratio_)
    print('pc1[:5]:', np.round(res['pc1'][:5], 3))
    print('kmeans type counts:',
          {t: int((res['k_type'] == t).sum()) for t in np.unique(res['k_type'])})
    print('final type counts:',
          {t: int((res['type'] == t).sum()) for t in np.unique(res['type'])})
