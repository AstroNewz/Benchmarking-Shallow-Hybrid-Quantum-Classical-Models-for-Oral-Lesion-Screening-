#!/usr/bin/env python3
"""
Fast globally entangled 16-qubit experiment (genuine single 16-qubit register,
NOT block-separable):
  U_enc^(16)(alpha) = RY(pi/4)^{\otimes 16} * CZ_ring^(16) * RY(alpha)^{\otimes 16}
Evaluates:
  - Fixed Q16 (global 16-qubit ring, 32 observables: 16 <Z_j> + 16 <Z_j Z_{(j+1)%16}>)
    vs. matched PCA-16 Logistic
  - Fidelity Q16 QSVM (exact 16-qubit state overlap |<psi(x)|psi(x')>|^2)
    vs. matched PCA-16 RBF-SVM
on:
  1. 5-fold patient-grouped CV on full SMART-OM corpus (N=2,436, 328 pts)
  2. 5-fold patient-grouped CV on A_lesion_polygon (N=316, 160 pts)
  3. Full-corpus held-out test (N=381, 48 pts) & polygon held-out test (N=49, 23 pts)

Memory-safe batched float32 statevector evolution (dim = 2^16 = 65,536).
Outputs: paper/q16_global.json
"""

import json
import math
import os
import sys
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paper.run_revision_experiments import (
    SEED,
    fit_predict_logistic,
    fit_predict_rbf_svm,
    load_data,
    patient_bootstrap_metrics,
)
from paper.run_pca32_and_fulltest import fit_predict_precomputed_svm

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q16_global.json")
N_Q = 16
DIM = 1 << N_Q  # 65,536


class MobileNetPCA16:
    def __init__(self, n_components=16, angle_max=np.pi / 2.0):
        self.n_components = n_components
        self.angle_max = angle_max

    def fit(self, X):
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self.pca = PCA(n_components=self.n_components, random_state=SEED).fit(Xn)
        Z = self.pca.transform(Xn)
        self.q02 = np.percentile(Z, 2, axis=0)
        self.q98 = np.percentile(Z, 98, axis=0)
        return self

    def transform_pca(self, X):
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return self.pca.transform(Xn)

    def transform_angles(self, X):
        Z = self.transform_pca(X)
        span = np.maximum(self.q98 - self.q02, 1e-12)
        return np.clip((Z - self.q02) / span * self.angle_max, 0.0, self.angle_max)


def _cz_ring_16():
    idx = np.arange(DIM, dtype=np.uint32)
    phase = np.ones(DIM, dtype=np.float32)
    for q in range(N_Q):
        q2 = (q + 1) % N_Q
        both = ((idx >> q) & 1) & ((idx >> q2) & 1)
        phase[both.astype(bool)] *= -1.0
    return phase


def _z_signs_16():
    idx = np.arange(DIM, dtype=np.uint32)
    z1 = np.array([np.where((idx >> q) & 1, -1.0, 1.0) for q in range(N_Q)], dtype=np.float32)
    zz = np.array([z1[q] * z1[(q + 1) % N_Q] for q in range(N_Q)], dtype=np.float32)
    return np.vstack([z1, zz])  # (32, 65536)


def _apply_ry_f32(states, angles):
    """In-place friendly vectorized RY on (B, 65536) float32 states."""
    B = states.shape[0]
    s = states.reshape((B,) + (2,) * N_Q)
    if angles.ndim == 1:
        angles = np.broadcast_to(angles[None, :], (B, N_Q))
    for q in range(N_Q):
        half = (angles[:, q] * 0.5).astype(np.float32)
        c = np.cos(half).reshape((B,) + (1,) * (N_Q - 1))
        sn = np.sin(half).reshape((B,) + (1,) * (N_Q - 1))
        ax = N_Q - q
        s = np.moveaxis(s, ax, -1)
        a0 = s[..., 0].copy()
        a1 = s[..., 1].copy()
        s[..., 0] = c * a0 - sn * a1
        s[..., 1] = sn * a0 + c * a1
        s = np.moveaxis(s, -1, ax)
    return s.reshape(B, DIM)


def q16_states_and_features(angles, cz16, obs32, batch_size=64, return_states=True):
    """Exact globally entangled 16-qubit states: RY(pi/4) * CZ_ring^(16) * RY(alpha) |0>^16."""
    N = angles.shape[0]
    feats = np.empty((N, 32), dtype=np.float64)
    states_all = np.empty((N, DIM), dtype=np.float32) if return_states else None
    ry_mix = np.full(N_Q, np.pi / 4.0, dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        st = np.zeros((B, DIM), dtype=np.float32)
        st[:, 0] = 1.0
        st = _apply_ry_f32(st, angles[start:end])
        st *= cz16[None, :]
        st = _apply_ry_f32(st, ry_mix)
        probs = st * st
        feats[start:end] = probs @ obs32.T
        if return_states:
            states_all[start:end] = st
    return states_all, feats


def run_cv_16(X_c7, X_c6, y, pids, cz16, obs32, label):
    sgkf = StratifiedGroupKFold(n_splits=5)
    arms = ["c7", "pca16_log", "pca16_rbf", "q16_fixed", "q16_qsvm"]
    oof = {a: np.zeros(len(y)) for a in arms}
    for ti, vi in sgkf.split(X_c7, y, pids):
        sc7 = StandardScaler().fit(X_c7[ti])
        oof["c7"][vi] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            sc7.transform(X_c7[ti]), y[ti]).decision_function(sc7.transform(X_c7[vi]))

        pre = MobileNetPCA16().fit(X_c6[ti])
        p_tr, p_vi = pre.transform_pca(X_c6[ti]), pre.transform_pca(X_c6[vi])
        a_tr, a_vi = pre.transform_angles(X_c6[ti]), pre.transform_angles(X_c6[vi])

        scp = StandardScaler().fit(p_tr)
        oof["pca16_log"][vi] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scp.transform(p_tr), y[ti]).decision_function(scp.transform(p_vi))
        oof["pca16_rbf"][vi] = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=SEED).fit(
            scp.transform(p_tr), y[ti]).decision_function(scp.transform(p_vi))

        st_tr, f_tr = q16_states_and_features(a_tr, cz16, obs32, batch_size=64, return_states=True)
        st_vi, f_vi = q16_states_and_features(a_vi, cz16, obs32, batch_size=64, return_states=True)

        scq = StandardScaler().fit(f_tr)
        oof["q16_fixed"][vi] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scq.transform(f_tr), y[ti]).decision_function(scq.transform(f_vi))

        K_tr = (st_tr @ st_tr.T).astype(np.float64) ** 2
        K_vi = (st_vi @ st_tr.T).astype(np.float64) ** 2
        oof["q16_qsvm"][vi] = SVC(C=1.0, kernel="precomputed", class_weight="balanced", random_state=SEED).fit(
            K_tr, y[ti]).decision_function(K_vi)
        del st_tr, st_vi, K_tr, K_vi

    pairs = [("q16_fixed", "c7"), ("q16_fixed", "pca16_log"),
             ("q16_qsvm", "c7"), ("q16_qsvm", "pca16_rbf")]
    return patient_bootstrap_metrics(y, pids, oof, pairs, n_draws=2000, seed=SEED)


def run_test_16(X7_tr, X6_tr, y_tr, p_tr, X7_te, X6_te, y_te, p_te, cz16, obs32):
    scores = {}
    scores["c7"], _ = fit_predict_logistic(X7_tr, y_tr, p_tr, X7_te)
    pre = MobileNetPCA16().fit(X6_tr)
    p_tr_16, p_te_16 = pre.transform_pca(X6_tr), pre.transform_pca(X6_te)
    a_tr_16, a_te_16 = pre.transform_angles(X6_tr), pre.transform_angles(X6_te)

    scores["pca16_log"], _ = fit_predict_logistic(p_tr_16, y_tr, p_tr, p_te_16)
    scores["pca16_rbf"], _ = fit_predict_rbf_svm(p_tr_16, y_tr, p_tr, p_te_16)

    st_tr, f_tr = q16_states_and_features(a_tr_16, cz16, obs32, batch_size=64, return_states=True)
    st_te, f_te = q16_states_and_features(a_te_16, cz16, obs32, batch_size=64, return_states=True)
    scores["q16_fixed"], _ = fit_predict_logistic(f_tr, y_tr, p_tr, f_te)
    K_tr = (st_tr @ st_tr.T).astype(np.float64) ** 2
    K_te = (st_te @ st_tr.T).astype(np.float64) ** 2
    scores["q16_qsvm"], _ = fit_predict_precomputed_svm(K_tr, y_tr, p_tr, K_te)
    pairs = [("q16_fixed", "c7"), ("q16_fixed", "pca16_log"),
             ("q16_qsvm", "c7"), ("q16_qsvm", "pca16_rbf")]
    return patient_bootstrap_metrics(y_te, p_te, scores, pairs, n_draws=2000, seed=SEED)


def main():
    t0 = time.time()
    D = load_data()
    cz16 = _cz_ring_16()
    obs32 = _z_signs_16()
    m_poly = (D["roi_src"] == "lesion_polygon")
    m_tr = (D["parts"] == "train")
    m_te = (D["parts"] == "test")

    print("1/4 Globally entangled 16-qubit 5-fold CV on full corpus (N=2,436, 328 pts) ...")
    cv_full = run_cv_16(D["X_c7"], D["X_c6"], D["y"], D["pids"], cz16, obs32, "full_2436")

    print("2/4 Globally entangled 16-qubit 5-fold CV on polygon subset (N=316, 160 pts) ...")
    cv_poly = run_cv_16(D["X_c7"][m_poly], D["X_c6"][m_poly], D["y"][m_poly], D["pids"][m_poly], cz16, obs32, "poly_316")

    print("3/4 Globally entangled 16-qubit FULL held-out test (N=381, 48 pts) ...")
    te_full = run_test_16(D["X_c7"][m_tr], D["X_c6"][m_tr], D["y"][m_tr], D["pids"][m_tr],
                          D["X_c7"][m_te], D["X_c6"][m_te], D["y"][m_te], D["pids"][m_te], cz16, obs32)

    print("4/4 Globally entangled 16-qubit POLYGON held-out test (N=49, 23 pts) ...")
    pm_tr, pm_te = m_poly & m_tr, m_poly & m_te
    te_poly = run_test_16(D["X_c7"][pm_tr], D["X_c6"][pm_tr], D["y"][pm_tr], D["pids"][pm_tr],
                          D["X_c7"][pm_te], D["X_c6"][pm_te], D["y"][pm_te], D["pids"][pm_te], cz16, obs32)

    out = {"runtime_seconds": round(time.time() - t0, 2),
           "cv_2436": cv_full, "cv_316": cv_poly, "test_381": te_full, "test_49": te_poly}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE in {out['runtime_seconds']}s")
    for title, b in [("CV 2436", cv_full), ("CV 316", cv_poly), ("TEST 381", te_full), ("TEST 49", te_poly)]:
        print(f"\n=== {title} ===")
        for k, v in b["point"].items():
            ci = b["ci_pr"][k]
            print(f"  {k:12s} PR={v['pr_auc']:.4f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  ROC={v['roc_auc']:.4f}")
        for k, v in b["paired_pr"].items():
            print(f"    {k:26s} d={v['delta']:+.4f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
