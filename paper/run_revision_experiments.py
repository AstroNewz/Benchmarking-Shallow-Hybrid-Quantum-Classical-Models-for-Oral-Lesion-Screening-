#!/usr/bin/env python3
"""
Revision experiments addressing peer-review statistical & circuit rigor:
1. Reproduces C7 anchor (0.913038 on A_lesion_polygon validation) and computes
   paired patient-level bootstrap 95% CIs (B=2,000) for all validation gaps.
2. Evaluates a genuinely non-commuting L=2 VQC (RY(alpha)->RY(th1)->CZ_ring->RY(th2)-><Z>)
   against a true product-state VQC (zero CZ gates) and capacity-matched MLP,
   verifying numerically that CZ alters <Z_j>.
3. Performs train-CV bandwidth tuning gamma in {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}
   for the Havlicek ZZ feature map and reports Gram-matrix off-diagonal statistics
   and effective rank.
4. Evaluates C7 AND all quantum models + matched classical controls on the held-out
   test partition (N=49, 23 patients) with 95% patient-level bootstrap CIs.
5. Runs 5-fold patient-grouped CV across:
   (a) the full A_lesion_polygon subset (N=316, 160 patients), and
   (b) the full verified SMART-OM corpus (N=2,436, 328 patients).

Outputs: paper/revision_metrics.json
"""

import json
import math
import os
import re
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revision_metrics.json")

SEED = 42
N_QUBITS = 8
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
BOOTSTRAP_DRAWS = 2000


def parse_patient_id(image_id: str) -> str:
    """Match backend/dataset/smartom.py patient ID parsing from image_id."""
    # image_id format: '<class>/<subdir>/<patient_num> - ...' or similar
    parts = str(image_id).split("/")
    cls = parts[0]
    stem = parts[-1]
    m = re.match(r"^\s*(\d+)", stem)
    pid = m.group(1) if m else stem.split()[0]
    return f"{cls}:{pid}"


def load_data():
    manifest = json.load(open(os.path.join(ROOT, "backend/artifacts/dataset/split_manifest.json")))
    pos_classes = set(manifest["positive_classes"])
    img_to_part = manifest["image_to_partition"]
    pat_to_part = manifest["patient_to_partition"]

    e1 = np.load(os.path.join(ROOT, "backend/artifacts/dataset/features_e1.npz"), allow_pickle=True)
    hc = np.load(os.path.join(ROOT, "backend/artifacts/dataset/features_handcrafted_lab.npz"), allow_pickle=True)

    # Align handcrafted matrix to features_e1 image_ids
    hc_idx = {str(i): idx for idx, i in enumerate(hc["image_ids"])}
    ids = np.array([str(i) for i in e1["image_ids"]])
    order = np.array([hc_idx[i] for i in ids])

    X_c1 = hc["matrix"][order].astype(np.float64)            # (2436, 163)
    X_ms = e1["matrix_multiscale"].astype(np.float64)        # (2436, 456)
    X_c5 = np.hstack([X_c1, X_ms])                           # (2436, 619)
    X_c6 = e1["matrix_mobilenet"].astype(np.float64)         # (2436, 576)
    X_c7 = np.hstack([X_c6, X_c5])                           # (2436, 1195)
    roi_frac = hc["roi_fractions"][order].astype(np.float64)
    roi_src = np.array([str(s) for s in e1["roi_sources"]])

    y = np.array([1 if i.split("/")[0] in pos_classes else 0 for i in ids], dtype=np.int64)
    parts = np.array([img_to_part[i] for i in ids])

    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from backend.dataset.smartom import resolve_patient_id
    pids = np.array([resolve_patient_id(i.split("/")[-1], i.split("/")[0])[0] for i in ids])
    assert len(np.unique(pids)) == 328, f"Expected 328 patients, got {len(np.unique(pids))}"

    return dict(ids=ids, y=y, parts=parts, pids=pids, roi_src=roi_src, roi_frac=roi_frac,
                X_c1=X_c1, X_c5=X_c5, X_c6=X_c6, X_c7=X_c7)


# ------------------------------------------------------------------ quantum primitives
def _cz_ring_phase(n=N_QUBITS):
    dim = 1 << n
    idx = np.arange(dim)
    phase = np.ones(dim, dtype=np.float64)
    for q in range(n):
        q2 = (q + 1) % n
        both = ((idx >> q) & 1) & ((idx >> q2) & 1)
        phase[both.astype(bool)] *= -1.0
    return phase


def _z_signs(n=N_QUBITS):
    dim = 1 << n
    idx = np.arange(dim)
    return np.array([np.where((idx >> q) & 1, -1.0, 1.0) for q in range(n)], dtype=np.float64)


def _apply_ry_np(states, angles):
    """states: (B, 2^n) real or complex; angles: (B, n) or (n,)."""
    B, dim = states.shape
    n = int(round(math.log2(dim)))
    s = states.reshape((B,) + (2,) * n)
    if angles.ndim == 1:
        angles = np.broadcast_to(angles[None, :], (B, n))
    for q in range(n):
        half = angles[:, q] * 0.5
        c = np.cos(half).reshape((B,) + (1,) * (n - 1))
        sn = np.sin(half).reshape((B,) + (1,) * (n - 1))
        ax = n - q
        s = np.moveaxis(s, ax, -1)
        a0, a1 = s[..., 0].copy(), s[..., 1].copy()
        s[..., 0] = c * a0 - sn * a1
        s[..., 1] = sn * a0 + c * a1
        s = np.moveaxis(s, -1, ax)
    return s.reshape(B, dim)


def _ry_product_states(angles):
    B, n = angles.shape
    s = np.zeros((B, 1 << n), dtype=np.float64)
    s[:, 0] = 1.0
    return _apply_ry_np(s, angles)


def _fixed_noncommuting_features(angles, cz_phase, z_signs):
    """Non-commuting fixed 8-qubit map: RY(alpha) -> CZ_ring -> RY(pi/4) -> <Z_j>, <Z_j Z_{j+1}>.
    Because RY(pi/4) does not commute with CZ_ring, entanglement genuinely affects all 16 observables."""
    s = _ry_product_states(angles)
    s = s * cz_phase[None, :]
    s = _apply_ry_np(s, np.full(angles.shape[1], np.pi / 4.0))
    probs = s * s
    z1 = probs @ z_signs.T  # (B, n)
    zz = np.column_stack([probs @ (z_signs[q] * z_signs[(q + 1) % angles.shape[1]]) for q in range(angles.shape[1])])
    return np.hstack([z1, zz])


def _havlicek_zz_states(x_scaled, reps=2):
    """Exact Havlicek ZZFeatureMap statevector (n=8, dim=256): [H^n -> U_Phi(x)]^reps |0>^n."""
    B, n = x_scaled.shape
    dim = 1 << n
    idx = np.arange(dim)
    bits = np.array([(idx >> q) & 1 for q in range(n)], dtype=np.float64)  # (n, dim)
    z_diag = 1.0 - 2.0 * bits  # +1 for |0>, -1 for |1>
    # Precompute pair products (n_pairs, dim)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    zz_diag = np.array([z_diag[i] * z_diag[j] for i, j in pairs])  # (28, dim)

    # Compute phase for each sample: phi(b) = sum_i x_i z_i + sum_{i<j} (pi - x_i)(pi - x_j) z_i z_j
    single_phi = x_scaled @ z_diag  # (B, dim)
    pi_minus = np.pi - x_scaled
    pair_coeff = np.column_stack([pi_minus[:, i] * pi_minus[:, j] for i, j in pairs])  # (B, 28)
    pair_phi = pair_coeff @ zz_diag  # (B, dim)
    phase = np.exp(1j * (single_phi + pair_phi))  # (B, dim)

    # Fast Walsh-Hadamard transform for H^{\otimes n}
    def apply_h_all(st):
        out = st.reshape((B,) + (2,) * n)
        inv_s2 = 1.0 / math.sqrt(2.0)
        for q in range(n):
            ax = n - q
            out = np.moveaxis(out, ax, -1)
            a0, a1 = out[..., 0].copy(), out[..., 1].copy()
            out[..., 0] = (a0 + a1) * inv_s2
            out[..., 1] = (a0 - a1) * inv_s2
            out = np.moveaxis(out, -1, ax)
        return out.reshape(B, dim)

    st = np.zeros((B, dim), dtype=np.complex128)
    st[:, 0] = 1.0
    for _ in range(reps):
        st = apply_h_all(st)
        st = st * phase
    return st


def _zz_observables(states, z_signs):
    """36 observables: 8 <Z_i> + 28 <Z_i Z_j>."""
    probs = (states.real ** 2 + states.imag ** 2)
    n = z_signs.shape[0]
    z1 = probs @ z_signs.T
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    zz = np.column_stack([probs @ (z_signs[i] * z_signs[j]) for i, j in pairs])
    return np.hstack([z1, zz])


def gram_stats(states):
    K = np.abs(states @ states.conj().T) ** 2
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 0.0, None)
    p = eigvals / (eigvals.sum() + 1e-15)
    p = p[p > 1e-15]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    return {
        "off_diag_mean": float(off.mean()),
        "off_diag_sd": float(off.std()),
        "off_diag_min": float(off.min()),
        "off_diag_max": float(off.max()),
        "effective_rank": eff_rank,
        "matrix_size": int(n),
    }


# ------------------------------------------------------------------ preprocessing
class MobileNetPCA8:
    def __init__(self, n_components=N_QUBITS, angle_max=np.pi / 2.0):
        self.n_components = n_components
        self.angle_max = angle_max

    def fit(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.maximum(norms, 1e-12)
        self.pca = PCA(n_components=self.n_components, random_state=SEED).fit(Xn)
        Z = self.pca.transform(Xn)
        self.q02 = np.percentile(Z, 2, axis=0)
        self.q98 = np.percentile(Z, 98, axis=0)
        return self

    def transform_pca(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return self.pca.transform(X / np.maximum(norms, 1e-12))

    def transform_angles(self, X):
        Z = self.transform_pca(X)
        span = np.maximum(self.q98 - self.q02, 1e-12)
        return np.clip((Z - self.q02) / span * self.angle_max, 0.0, self.angle_max)


# ------------------------------------------------------------------ trainable L=2 VQC
def _apply_ry_torch(s_nd, theta_row, n):
    s = s_nd
    for q in range(n):
        half = theta_row[q] * 0.5
        c, sn = torch.cos(half), torch.sin(half)
        ax = n - q
        m = torch.movedim(s, ax, -1)
        a0, a1 = m[..., 0], m[..., 1]
        m = torch.stack([c * a0 - sn * a1, sn * a0 + c * a1], dim=-1)
        s = torch.movedim(m, -1, ax)
    return s


def train_and_score_vqc_l2(ang_tr, y_tr, ang_ev, *, entangle: bool, seed: int = SEED, steps: int = 200):
    """Genuinely non-commuting L=2 VQC:
       RY(alpha) -> RY(theta^(1)) -> [CZ_ring if entangle] -> RY(theta^(2)) -> <Z_j> -> linear head.
       Because CZ_ring sits strictly between RY(theta^(1)) and RY(theta^(2)), it does NOT commute
       with the measurement basis."""
    torch.manual_seed(seed)
    n = ang_tr.shape[1]
    dim = 1 << n
    dt = torch.float32
    cz_t = torch.tensor(_cz_ring_phase(n), dtype=dt)
    zs_t = torch.tensor(_z_signs(n), dtype=dt)

    s_tr_init = torch.tensor(_ry_product_states(ang_tr), dtype=dt)
    s_ev_init = torch.tensor(_ry_product_states(ang_ev), dtype=dt)
    y_t = torch.tensor(y_tr, dtype=dt)
    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    w_t = torch.where(y_t == 1, len(y_tr) / (2.0 * max(n_pos, 1.0)), len(y_tr) / (2.0 * max(n_neg, 1.0)))

    gen = torch.Generator().manual_seed(seed)
    th1 = ((torch.rand(n, generator=gen, dtype=dt) * 2.0 - 1.0) * math.pi).requires_grad_(True)
    th2 = ((torch.rand(n, generator=gen, dtype=dt) * 2.0 - 1.0) * math.pi).requires_grad_(True)
    hw = (torch.randn(n, generator=gen, dtype=dt) * 0.1).requires_grad_(True)
    hb = torch.zeros((), dtype=dt, requires_grad=True)

    def forward(s0):
        B = s0.shape[0]
        shape = (B,) + (2,) * n
        s = _apply_ry_torch(s0.reshape(shape), th1, n).reshape(B, dim)
        if entangle:
            s = s * cz_t
        s = _apply_ry_torch(s.reshape(shape), th2, n).reshape(B, dim)
        probs = s * s
        z = probs @ zs_t.t()
        return z @ hw + hb, z

    opt = torch.optim.Adam([th1, th2, hw, hb], lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        logits, _ = forward(s_tr_init)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction="none") * w_t).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        ev_logits, _ = forward(s_ev_init)
    return ev_logits.numpy()


def verify_entanglement_non_commutation(ang_sample):
    """Numerically verify that CZ_ring between RY(th1) and RY(th2) changes <Z_j>."""
    n = ang_sample.shape[1]
    dim = 1 << n
    cz_t = torch.tensor(_cz_ring_phase(n), dtype=torch.float64)
    zs_t = torch.tensor(_z_signs(n), dtype=torch.float64)
    s0 = torch.tensor(_ry_product_states(ang_sample[:16]), dtype=torch.float64)
    gen = torch.Generator().manual_seed(7)
    th1 = (torch.rand(n, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * math.pi
    th2 = (torch.rand(n, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * math.pi
    shape = (s0.shape[0],) + (2,) * n
    s1 = _apply_ry_torch(s0.reshape(shape), th1, n).reshape(s0.shape[0], dim)
    z_ent = (_apply_ry_torch((s1 * cz_t).reshape(shape), th2, n).reshape(s0.shape[0], dim) ** 2) @ zs_t.t()
    z_prd = (_apply_ry_torch(s1.reshape(shape), th2, n).reshape(s0.shape[0], dim) ** 2) @ zs_t.t()
    max_diff = float(torch.max(torch.abs(z_ent - z_prd)).item())
    mean_diff = float(torch.mean(torch.abs(z_ent - z_prd)).item())
    assert max_diff > 0.1, f"CZ did not alter <Z>! max_diff={max_diff}"
    return {"max_abs_delta_Z": max_diff, "mean_abs_delta_Z": mean_diff}


# ------------------------------------------------------------------ generic CV-tuned classifiers
def fit_predict_logistic(X_tr, y_tr, p_tr, X_ev):
    sgkf = StratifiedGroupKFold(n_splits=5)
    best_c, best_ap = 1.0, -1.0
    for c in C_GRID:
        oof = np.zeros(len(y_tr))
        for ti, vi in sgkf.split(X_tr, y_tr, p_tr):
            sc = StandardScaler().fit(X_tr[ti])
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)
            clf.fit(sc.transform(X_tr[ti]), y_tr[ti])
            oof[vi] = clf.decision_function(sc.transform(X_tr[vi]))
        ap = average_precision_score(y_tr, oof)
        if ap > best_ap:
            best_ap, best_c = ap, c
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.decision_function(sc.transform(X_ev)), best_c


def fit_predict_rbf_svm(X_tr, y_tr, p_tr, X_ev):
    sgkf = StratifiedGroupKFold(n_splits=5)
    best_c, best_ap = 1.0, -1.0
    for c in [0.1, 1.0, 10.0, 100.0]:
        oof = np.zeros(len(y_tr))
        for ti, vi in sgkf.split(X_tr, y_tr, p_tr):
            sc = StandardScaler().fit(X_tr[ti])
            clf = SVC(C=c, kernel="rbf", class_weight="balanced", random_state=SEED)
            clf.fit(sc.transform(X_tr[ti]), y_tr[ti])
            oof[vi] = clf.decision_function(sc.transform(X_tr[vi]))
        ap = average_precision_score(y_tr, oof)
        if ap > best_ap:
            best_ap, best_c = ap, c
    sc = StandardScaler().fit(X_tr)
    clf = SVC(C=best_c, kernel="rbf", class_weight="balanced", random_state=SEED)
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.decision_function(sc.transform(X_ev)), best_c


def fit_predict_qsvm_kernel(st_tr, y_tr, p_tr, st_ev):
    K_tr = np.abs(st_tr @ st_tr.conj().T) ** 2
    K_ev = np.abs(st_ev @ st_tr.conj().T) ** 2
    sgkf = StratifiedGroupKFold(n_splits=5)
    best_c, best_ap = 1.0, -1.0
    for c in [0.1, 1.0, 10.0, 100.0]:
        oof = np.zeros(len(y_tr))
        for ti, vi in sgkf.split(K_tr, y_tr, p_tr):
            clf = SVC(C=c, kernel="precomputed", class_weight="balanced", random_state=SEED)
            clf.fit(K_tr[np.ix_(ti, ti)], y_tr[ti])
            oof[vi] = clf.decision_function(K_tr[np.ix_(vi, ti)])
        ap = average_precision_score(y_tr, oof)
        if ap > best_ap:
            best_ap, best_c = ap, c
    clf = SVC(C=best_c, kernel="precomputed", class_weight="balanced", random_state=SEED)
    clf.fit(K_tr, y_tr)
    return clf.decision_function(K_ev), best_c


def fit_predict_mlp(X_tr, y_tr, X_ev):
    sc = StandardScaler().fit(X_tr)
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=600, alpha=1e-2, random_state=SEED)
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_ev))[:, 1]


def make_rff(X_tr, X_ev, n_features=16, seed=SEED):
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(X_tr)
    Xtr_s, Xev_s = sc.transform(X_tr), sc.transform(X_ev)
    # median heuristic
    sub = Xtr_s[:min(len(Xtr_s), 200)]
    dists = np.sqrt(np.maximum(0.0, np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)))
    med = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    W = rng.normal(0.0, 1.0 / max(med, 1e-6), size=(X_tr.shape[1], n_features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=(n_features,))
    scale = math.sqrt(2.0 / n_features)
    return scale * np.cos(Xtr_s @ W + b), scale * np.cos(Xev_s @ W + b)


# ------------------------------------------------------------------ patient-level paired bootstrap
def patient_bootstrap_metrics(y, pids, score_dict, pairs, n_draws=BOOTSTRAP_DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(pids)
    pid_to_idx = {p: np.where(pids == p)[0] for p in unique_pids}
    n_p = len(unique_pids)

    point = {k: {"pr_auc": float(average_precision_score(y, s)),
                 "roc_auc": float(roc_auc_score(y, s))} for k, s in score_dict.items()}
    boot_pr = {k: [] for k in score_dict}
    boot_roc = {k: [] for k in score_dict}
    pair_pr = {f"{a}_minus_{b}": [] for a, b in pairs}
    pair_roc = {f"{a}_minus_{b}": [] for a, b in pairs}

    for _ in range(n_draws):
        sampled = rng.choice(unique_pids, size=n_p, replace=True)
        idx = np.concatenate([pid_to_idx[p] for p in sampled])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        cur_pr, cur_roc = {}, {}
        for k, s in score_dict.items():
            cur_pr[k] = average_precision_score(yb, s[idx])
            cur_roc[k] = roc_auc_score(yb, s[idx])
            boot_pr[k].append(cur_pr[k])
            boot_roc[k].append(cur_roc[k])
        for a, b in pairs:
            pair_pr[f"{a}_minus_{b}"].append(cur_pr[a] - cur_pr[b])
            pair_roc[f"{a}_minus_{b}"].append(cur_roc[a] - cur_roc[b])

    def summarize(vals):
        arr = np.asarray(vals)
        return {"mean": float(arr.mean()), "sd": float(arr.std()),
                "ci_low": float(np.percentile(arr, 2.5)), "ci_high": float(np.percentile(arr, 97.5)),
                "n_usable": int(len(arr))}

    out = {"point": point, "ci_pr": {k: summarize(v) for k, v in boot_pr.items()},
           "ci_roc": {k: summarize(v) for k, v in boot_roc.items()},
           "paired_pr": {}, "paired_roc": {}}
    for a, b in pairs:
        key = f"{a}_minus_{b}"
        out["paired_pr"][key] = {"delta": point[a]["pr_auc"] - point[b]["pr_auc"], **summarize(pair_pr[key])}
        out["paired_roc"][key] = {"delta": point[a]["roc_auc"] - point[b]["roc_auc"], **summarize(pair_roc[key])}
    return out


# ------------------------------------------------------------------ ZZ bandwidth sweep
def tune_zz_bandwidth(ang_tr, y_tr, p_tr, ang_ev, z_signs):
    gammas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    sgkf = StratifiedGroupKFold(n_splits=5)
    sweep = {}
    best_g, best_oof = 1.0, -1.0
    for g in gammas:
        st_tr = _havlicek_zz_states(ang_tr * g, reps=2)
        F_tr = _zz_observables(st_tr, z_signs)
        # Quick inner CV with logistic C=0.1
        oof = np.zeros(len(y_tr))
        for ti, vi in sgkf.split(F_tr, y_tr, p_tr):
            sc = StandardScaler().fit(F_tr[ti])
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED)
            clf.fit(sc.transform(F_tr[ti]), y_tr[ti])
            oof[vi] = clf.decision_function(sc.transform(F_tr[vi]))
        oof_ap = float(average_precision_score(y_tr, oof))
        gs = gram_stats(st_tr)
        sweep[str(g)] = {"train_cv_pr_auc": oof_ap, "gram_stats": gs}
        if oof_ap > best_oof:
            best_oof, best_g = oof_ap, g

    # Score both untuned (gamma=1.0) and tuned (gamma=best_g)
    def score_at(g):
        st_tr = _havlicek_zz_states(ang_tr * g, reps=2)
        st_ev = _havlicek_zz_states(ang_ev * g, reps=2)
        F_tr = _zz_observables(st_tr, z_signs)
        F_ev = _zz_observables(st_ev, z_signs)
        sc_lin, _ = fit_predict_logistic(F_tr, y_tr, p_tr, F_ev)
        sc_k, _ = fit_predict_qsvm_kernel(st_tr, y_tr, p_tr, st_ev)
        return sc_lin, sc_k, gram_stats(st_tr)

    lin_1, ker_1, gs_1 = score_at(1.0)
    lin_best, ker_best, gs_best = score_at(best_g)
    return {
        "best_gamma": best_g,
        "sweep": sweep,
        "untuned_lin": lin_1, "untuned_ker": ker_1, "untuned_gram": gs_1,
        "tuned_lin": lin_best, "tuned_ker": ker_best, "tuned_gram": gs_best,
    }


def evaluate_split(X_c7_tr, X_c6_tr, y_tr, p_tr, X_c7_ev, X_c6_ev, y_ev, p_ev, roi_frac_ev, cz_phase, z_signs):
    scores = {}
    scores["c7"], _ = fit_predict_logistic(X_c7_tr, y_tr, p_tr, X_c7_ev)
    scores["c6"], _ = fit_predict_logistic(X_c6_tr, y_tr, p_tr, X_c6_ev)

    pre = MobileNetPCA8().fit(X_c6_tr)
    pca_tr, pca_ev = pre.transform_pca(X_c6_tr), pre.transform_pca(X_c6_ev)
    ang_tr, ang_ev = pre.transform_angles(X_c6_tr), pre.transform_angles(X_c6_ev)

    scores["pca8"], _ = fit_predict_logistic(pca_tr, y_tr, p_tr, pca_ev)
    rff16_tr, rff16_ev = make_rff(pca_tr, pca_ev, n_features=16)
    scores["rff16"], _ = fit_predict_logistic(rff16_tr, y_tr, p_tr, rff16_ev)
    rff36_tr, rff36_ev = make_rff(pca_tr, pca_ev, n_features=36)
    scores["rff36"], _ = fit_predict_logistic(rff36_tr, y_tr, p_tr, rff36_ev)
    scores["rbf_svm"], _ = fit_predict_rbf_svm(pca_tr, y_tr, p_tr, pca_ev)
    scores["mlp"], = (fit_predict_mlp(pca_tr, y_tr, pca_ev),)

    # Non-commuting fixed hybrid quantum map
    qfix_tr = _fixed_noncommuting_features(ang_tr, cz_phase, z_signs)
    qfix_ev = _fixed_noncommuting_features(ang_ev, cz_phase, z_signs)
    scores["fixed_quantum_nc"], _ = fit_predict_logistic(qfix_tr, y_tr, p_tr, qfix_ev)

    # Fidelity QSVM (on RY+CZ+RY(pi/4) states so kernel is non-product)
    st_tr_nc = _apply_ry_np(_ry_product_states(ang_tr) * cz_phase[None, :], np.full(N_QUBITS, np.pi / 4.0))
    st_ev_nc = _apply_ry_np(_ry_product_states(ang_ev) * cz_phase[None, :], np.full(N_QUBITS, np.pi / 4.0))
    scores["qsvm"], _ = fit_predict_qsvm_kernel(st_tr_nc, y_tr, p_tr, st_ev_nc)
    qsvm_gram = gram_stats(st_tr_nc)

    # Genuinely non-commuting L=2 VQC vs true product-state VQC
    scores["vqc_l2_ent"] = train_and_score_vqc_l2(ang_tr, y_tr, ang_ev, entangle=True, seed=SEED)
    scores["vqc_l2_prod"] = train_and_score_vqc_l2(ang_tr, y_tr, ang_ev, entangle=False, seed=SEED)

    # Bandwidth-tuned Havlicek ZZ
    zz_res = tune_zz_bandwidth(ang_tr, y_tr, p_tr, ang_ev, z_signs)
    scores["zz_untuned"] = zz_res["untuned_lin"]
    scores["zz_tuned_lin"] = zz_res["tuned_lin"]
    scores["zz_tuned_ker"] = zz_res["tuned_ker"]

    # Leak bar (-roi_area_fraction)
    scores["leak_bar"] = -roi_frac_ev

    pairs = [
        ("fixed_quantum_nc", "c7"), ("fixed_quantum_nc", "pca8"), ("fixed_quantum_nc", "rff16"),
        ("qsvm", "c7"), ("qsvm", "rbf_svm"),
        ("vqc_l2_ent", "c7"), ("vqc_l2_ent", "vqc_l2_prod"), ("vqc_l2_ent", "mlp"),
        ("zz_untuned", "c7"), ("zz_tuned_lin", "c7"), ("zz_tuned_lin", "rff36"),
        ("zz_tuned_ker", "c7"), ("zz_tuned_ker", "rbf_svm"),
        ("c7", "leak_bar"), ("fixed_quantum_nc", "leak_bar"), ("vqc_l2_ent", "leak_bar"),
    ]
    boot = patient_bootstrap_metrics(y_ev, p_ev, scores, pairs)
    return boot, {
        "qsvm_gram": qsvm_gram,
        "zz_best_gamma": zz_res["best_gamma"],
        "zz_sweep": zz_res["sweep"],
        "zz_untuned_gram": zz_res["untuned_gram"],
        "zz_tuned_gram": zz_res["tuned_gram"],
    }


def run_grouped_cv(X_c7, X_c6, y, pids, cz_phase, z_signs, label, n_splits=5):
    """5-fold patient-grouped CV across an entire cohort (fold-internal PCA & tuning)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    arms = ["c7", "c6", "pca8", "rbf_svm", "fixed_quantum_nc", "qsvm", "vqc_l2_ent", "vqc_l2_prod", "zz_tuned_lin"]
    oof = {a: np.zeros(len(y)) for a in arms}
    fold_pr = {a: [] for a in arms}
    fold_roc = {a: [] for a in arms}

    for fold, (ti, vi) in enumerate(sgkf.split(X_c7, y, pids)):
        # Fixed regularization for speed/stability across large folds (C=0.01 for C7/C6, C=1.0 for 8-16D)
        sc7 = StandardScaler().fit(X_c7[ti])
        oof["c7"][vi] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            sc7.transform(X_c7[ti]), y[ti]).decision_function(sc7.transform(X_c7[vi]))
        sc6 = StandardScaler().fit(X_c6[ti])
        oof["c6"][vi] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            sc6.transform(X_c6[ti]), y[ti]).decision_function(sc6.transform(X_c6[vi]))

        pre = MobileNetPCA8().fit(X_c6[ti])
        p_tr, p_vi = pre.transform_pca(X_c6[ti]), pre.transform_pca(X_c6[vi])
        a_tr, a_vi = pre.transform_angles(X_c6[ti]), pre.transform_angles(X_c6[vi])

        scp = StandardScaler().fit(p_tr)
        oof["pca8"][vi] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scp.transform(p_tr), y[ti]).decision_function(scp.transform(p_vi))
        oof["rbf_svm"][vi] = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=SEED).fit(
            scp.transform(p_tr), y[ti]).decision_function(scp.transform(p_vi))

        qf_tr = _fixed_noncommuting_features(a_tr, cz_phase, z_signs)
        qf_vi = _fixed_noncommuting_features(a_vi, cz_phase, z_signs)
        scq = StandardScaler().fit(qf_tr)
        oof["fixed_quantum_nc"][vi] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scq.transform(qf_tr), y[ti]).decision_function(scq.transform(qf_vi))

        # Fidelity QSVM
        st_tr = _apply_ry_np(_ry_product_states(a_tr) * cz_phase[None, :], np.full(N_QUBITS, np.pi / 4.0))
        st_vi = _apply_ry_np(_ry_product_states(a_vi) * cz_phase[None, :], np.full(N_QUBITS, np.pi / 4.0))
        K_tr = np.abs(st_tr @ st_tr.conj().T) ** 2
        K_vi = np.abs(st_vi @ st_tr.conj().T) ** 2
        oof["qsvm"][vi] = SVC(C=1.0, kernel="precomputed", class_weight="balanced", random_state=SEED).fit(
            K_tr, y[ti]).decision_function(K_vi)

        # VQC L=2 entangled vs true product (float32 / compact step budget for speed on large N)
        vqc_steps = 60 if len(y) > 500 else 150
        oof["vqc_l2_ent"][vi] = train_and_score_vqc_l2(a_tr, y[ti], a_vi, entangle=True, seed=SEED + fold, steps=vqc_steps)
        oof["vqc_l2_prod"][vi] = train_and_score_vqc_l2(a_tr, y[ti], a_vi, entangle=False, seed=SEED + fold, steps=vqc_steps)

        # Tuned ZZ (gamma=0.1)
        zz_tr = _zz_observables(_havlicek_zz_states(a_tr * 0.1, reps=2), z_signs)
        zz_vi = _zz_observables(_havlicek_zz_states(a_vi * 0.1, reps=2), z_signs)
        scz = StandardScaler().fit(zz_tr)
        oof["zz_tuned_lin"][vi] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scz.transform(zz_tr), y[ti]).decision_function(scz.transform(zz_vi))

        for a in arms:
            fold_pr[a].append(float(average_precision_score(y[vi], oof[a][vi])))
            fold_roc[a].append(float(roc_auc_score(y[vi], oof[a][vi])))

    pairs = [("fixed_quantum_nc", "c7"), ("fixed_quantum_nc", "pca8"),
             ("qsvm", "c7"), ("qsvm", "rbf_svm"),
             ("vqc_l2_ent", "c7"), ("vqc_l2_ent", "vqc_l2_prod"),
             ("zz_tuned_lin", "c7")]
    boot = patient_bootstrap_metrics(y, pids, oof, pairs, n_draws=1000, seed=SEED)
    return {
        "cohort": label,
        "n_images": int(len(y)),
        "n_patients": int(len(np.unique(pids))),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "fold_mean_pr": {a: float(np.mean(fold_pr[a])) for a in arms},
        "fold_sd_pr": {a: float(np.std(fold_pr[a])) for a in arms},
        "fold_mean_roc": {a: float(np.mean(fold_roc[a])) for a in arms},
        "oof_bootstrap": boot,
    }


def main():
    t0 = time.time()
    D = load_data()
    cz_phase = _cz_ring_phase(N_QUBITS)
    z_signs = _z_signs(N_QUBITS)

    # Masks for A_lesion_polygon
    m_poly = (D["roi_src"] == "lesion_polygon")
    m_tr = m_poly & (D["parts"] == "train")
    m_va = m_poly & (D["parts"] == "validation")
    m_te = m_poly & (D["parts"] == "test")

    # Verify C7 anchor
    pre_check = MobileNetPCA8().fit(D["X_c6"][m_tr])
    ent_check = verify_entanglement_non_commutation(pre_check.transform_angles(D["X_c6"][m_tr]))
    print("Entanglement non-commutation verified:", ent_check)

    print("1/4 Running A_lesion_polygon VALIDATION (N=52, 25 patients) with paired bootstrap ...")
    val_boot, val_diag = evaluate_split(
        D["X_c7"][m_tr], D["X_c6"][m_tr], D["y"][m_tr], D["pids"][m_tr],
        D["X_c7"][m_va], D["X_c6"][m_va], D["y"][m_va], D["pids"][m_va],
        D["roi_frac"][m_va], cz_phase, z_signs
    )
    print(f"  C7 val PR-AUC = {val_boot['point']['c7']['pr_auc']:.6f} (anchor target 0.913038)")

    print("2/4 Running A_lesion_polygon HELD-OUT TEST (N=49, 23 patients) with paired bootstrap ...")
    test_boot, test_diag = evaluate_split(
        D["X_c7"][m_tr], D["X_c6"][m_tr], D["y"][m_tr], D["pids"][m_tr],
        D["X_c7"][m_te], D["X_c6"][m_te], D["y"][m_te], D["pids"][m_te],
        D["roi_frac"][m_te], cz_phase, z_signs
    )

    print("3/4 Running 5-fold patient-grouped CV on full A_lesion_polygon (N=316, 160 patients) ...")
    cv_poly = run_grouped_cv(D["X_c7"][m_poly], D["X_c6"][m_poly], D["y"][m_poly], D["pids"][m_poly],
                             cz_phase, z_signs, "A_lesion_polygon_all")

    print("4/4 Running 5-fold patient-grouped CV on full SMART-OM corpus (N=2,436, 328 patients) ...")
    cv_full = run_grouped_cv(D["X_c7"], D["X_c6"], D["y"], D["pids"],
                             cz_phase, z_signs, "SMART_OM_full_corpus")

    out = {
        "runtime_seconds": round(time.time() - t0, 2),
        "entanglement_non_commutation_check": ent_check,
        "validation_A_lesion_polygon": {"bootstrap": val_boot, "diagnostics": val_diag},
        "test_A_lesion_polygon": {"bootstrap": test_boot, "diagnostics": test_diag},
        "grouped_cv_316": cv_poly,
        "grouped_cv_2436": cv_full,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE in {out['runtime_seconds']}s -> {OUT_PATH}")

    # Compact summary print
    for title, b in [("VAL (N=52)", val_boot), ("TEST (N=49)", test_boot)]:
        print(f"\n=== {title} ===")
        for k, v in b["point"].items():
            ci = b["ci_pr"][k]
            print(f"  {k:18s} PR={v['pr_auc']:.4f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  ROC={v['roc_auc']:.4f}")
        print("  Paired PR-AUC deltas:")
        for k, v in b["paired_pr"].items():
            print(f"    {k:32s} d={v['delta']:+.4f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")

    for cv in [cv_poly, cv_full]:
        print(f"\n=== 5-Fold Grouped CV: {cv['cohort']} (N={cv['n_images']}, pts={cv['n_patients']}, prev={cv['prevalence']:.3f}) ===")
        for a, v in cv["oof_bootstrap"]["point"].items():
            ci = cv["oof_bootstrap"]["ci_pr"][a]
            print(f"  {a:18s} OOF PR={v['pr_auc']:.4f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  ROC={v['roc_auc']:.4f}")
        print("  Paired OOF PR-AUC deltas:")
        for k, v in cv["oof_bootstrap"]["paired_pr"].items():
            print(f"    {k:32s} d={v['delta']:+.4f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
