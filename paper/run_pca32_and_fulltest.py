#!/usr/bin/env python3
"""
Fast supplemental revision experiments:
1. PCA-32 (4 x 8-qubit block-encoded = 32-qubit non-commuting quantum feature map &
   block-fidelity kernel) vs. matched PCA-32 classical controls (Logistic, RBF-SVM)
   to isolate the PCA-8 compression bottleneck from quantum expressivity.
2. Held-out test evaluation on the FULL SMART-OM test split (N_test = 381 images,
   48 patients, 18 positive images) in addition to A_lesion_polygon (N_test = 49, 23 patients).

Outputs: paper/pca32_and_fulltest.json
"""

import json
import math
import os
import sys
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paper.run_revision_experiments import (
    SEED,
    C_GRID,
    MobileNetPCA8,
    _apply_ry_np,
    _cz_ring_phase,
    _fixed_noncommuting_features,
    _havlicek_zz_states,
    _ry_product_states,
    _z_signs,
    _zz_observables,
    fit_predict_logistic,
    fit_predict_qsvm_kernel,
    fit_predict_rbf_svm,
    load_data,
    patient_bootstrap_metrics,
    train_and_score_vqc_l2,
)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca32_and_fulltest.json")


class MobileNetPCA32:
    """Fold-train PCA-32 on L2-normalized MobileNetV3 (576-D) -> 4 blocks of 8 angles in [0, pi/2]."""
    def __init__(self, n_components=32, angle_max=np.pi / 2.0):
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


def block32_quantum_features(ang32, cz8, zs8):
    """4 disjoint 8-qubit registers (32 qubits total), each RY(alpha)->CZ_ring->RY(pi/4)->16 observables => 64-D."""
    blocks = [_fixed_noncommuting_features(ang32[:, b * 8:(b + 1) * 8], cz8, zs8) for b in range(4)]
    return np.hstack(blocks)


def block32_fidelity_kernel(ang32_tr, ang32_ev, cz8):
    """Exact 32-qubit fidelity kernel for 4-block tensor-product state: K(x,x') = prod_{b=0}^3 |<psi_b(x)|psi_b(x')>|^2."""
    K_tr = np.ones((len(ang32_tr), len(ang32_tr)), dtype=np.float64)
    K_ev = np.ones((len(ang32_ev), len(ang32_tr)), dtype=np.float64)
    for b in range(4):
        st_tr = _apply_ry_np(_ry_product_states(ang32_tr[:, b * 8:(b + 1) * 8]) * cz8[None, :], np.full(8, np.pi / 4.0))
        st_ev = _apply_ry_np(_ry_product_states(ang32_ev[:, b * 8:(b + 1) * 8]) * cz8[None, :], np.full(8, np.pi / 4.0))
        K_tr *= np.abs(st_tr @ st_tr.T) ** 2
        K_ev *= np.abs(st_ev @ st_tr.T) ** 2
    return K_tr, K_ev


def fit_predict_precomputed_svm(K_tr, y_tr, p_tr, K_ev):
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


def run_pca32_cv(X_c7, X_c6, y, pids, cz8, zs8, label):
    sgkf = StratifiedGroupKFold(n_splits=5)
    arms = ["c7", "pca32_log", "pca32_rbf", "q32_fixed", "q32_qsvm"]
    oof = {a: np.zeros(len(y)) for a in arms}
    for ti, vi in sgkf.split(X_c7, y, pids):
        sc7 = StandardScaler().fit(X_c7[ti])
        oof["c7"][vi] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            sc7.transform(X_c7[ti]), y[ti]).decision_function(sc7.transform(X_c7[vi]))

        pre32 = MobileNetPCA32().fit(X_c6[ti])
        p_tr, p_vi = pre32.transform_pca(X_c6[ti]), pre32.transform_pca(X_c6[vi])
        a_tr, a_vi = pre32.transform_angles(X_c6[ti]), pre32.transform_angles(X_c6[vi])

        scp = StandardScaler().fit(p_tr)
        oof["pca32_log"][vi] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scp.transform(p_tr), y[ti]).decision_function(scp.transform(p_vi))
        oof["pca32_rbf"][vi] = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=SEED).fit(
            scp.transform(p_tr), y[ti]).decision_function(scp.transform(p_vi))

        qf_tr = block32_quantum_features(a_tr, cz8, zs8)
        qf_vi = block32_quantum_features(a_vi, cz8, zs8)
        scq = StandardScaler().fit(qf_tr)
        oof["q32_fixed"][vi] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scq.transform(qf_tr), y[ti]).decision_function(scq.transform(qf_vi))

        K_tr, K_vi = block32_fidelity_kernel(a_tr, a_vi, cz8)
        oof["q32_qsvm"][vi] = SVC(C=1.0, kernel="precomputed", class_weight="balanced", random_state=SEED).fit(
            K_tr, y[ti]).decision_function(K_vi)

    pairs = [("q32_fixed", "c7"), ("q32_fixed", "pca32_log"),
             ("q32_qsvm", "c7"), ("q32_qsvm", "pca32_rbf"),
             ("pca32_log", "c7")]
    boot = patient_bootstrap_metrics(y, pids, oof, pairs, n_draws=2000, seed=SEED)
    return {"cohort": label, "n_images": int(len(y)), "n_patients": int(len(np.unique(pids))), "boot": boot}


def run_full_corpus_test(D, cz8, zs8):
    """Evaluate all key classical + 8-qubit + 32-qubit models on the FULL SMART-OM held-out test set
       (N_test = 381 images, 48 patients, 18 positive images), fitted strictly on TRAIN (1,692 images, 230 patients)."""
    m_tr = (D["parts"] == "train")
    m_te = (D["parts"] == "test")
    X7_tr, X7_te = D["X_c7"][m_tr], D["X_c7"][m_te]
    X6_tr, X6_te = D["X_c6"][m_tr], D["X_c6"][m_te]
    y_tr, y_te = D["y"][m_tr], D["y"][m_te]
    p_tr, p_te = D["pids"][m_tr], D["pids"][m_te]

    scores = {}
    scores["c7"], _ = fit_predict_logistic(X7_tr, y_tr, p_tr, X7_te)
    scores["c6"], _ = fit_predict_logistic(X6_tr, y_tr, p_tr, X6_te)

    # 8-qubit / PCA-8
    pre8 = MobileNetPCA8().fit(X6_tr)
    p8_tr, p8_te = pre8.transform_pca(X6_tr), pre8.transform_pca(X6_te)
    a8_tr, a8_te = pre8.transform_angles(X6_tr), pre8.transform_angles(X6_te)

    scores["pca8"], _ = fit_predict_logistic(p8_tr, y_tr, p_tr, p8_te)
    scores["rbf8"], _ = fit_predict_rbf_svm(p8_tr, y_tr, p_tr, p8_te)
    scores["q8_fixed"], _ = fit_predict_logistic(
        _fixed_noncommuting_features(a8_tr, cz8, zs8), y_tr, p_tr,
        _fixed_noncommuting_features(a8_te, cz8, zs8)
    )
    st8_tr = _apply_ry_np(_ry_product_states(a8_tr) * cz8[None, :], np.full(8, np.pi / 4.0))
    st8_te = _apply_ry_np(_ry_product_states(a8_te) * cz8[None, :], np.full(8, np.pi / 4.0))
    scores["q8_qsvm"], _ = fit_predict_qsvm_kernel(st8_tr, y_tr, p_tr, st8_te)
    scores["q8_vqc_ent"] = train_and_score_vqc_l2(a8_tr, y_tr, a8_te, entangle=True, seed=SEED, steps=80)
    scores["q8_vqc_prod"] = train_and_score_vqc_l2(a8_tr, y_tr, a8_te, entangle=False, seed=SEED, steps=80)
    scores["q8_zz_tuned"], _ = fit_predict_logistic(
        _zz_observables(_havlicek_zz_states(a8_tr * 0.1, reps=2), zs8), y_tr, p_tr,
        _zz_observables(_havlicek_zz_states(a8_te * 0.1, reps=2), zs8)
    )

    # 32-qubit block-encoded / PCA-32
    pre32 = MobileNetPCA32().fit(X6_tr)
    p32_tr, p32_te = pre32.transform_pca(X6_tr), pre32.transform_pca(X6_te)
    a32_tr, a32_te = pre32.transform_angles(X6_tr), pre32.transform_angles(X6_te)

    scores["pca32_log"], _ = fit_predict_logistic(p32_tr, y_tr, p_tr, p32_te)
    scores["pca32_rbf"], _ = fit_predict_rbf_svm(p32_tr, y_tr, p_tr, p32_te)
    scores["q32_fixed"], _ = fit_predict_logistic(
        block32_quantum_features(a32_tr, cz8, zs8), y_tr, p_tr,
        block32_quantum_features(a32_te, cz8, zs8)
    )
    K32_tr, K32_te = block32_fidelity_kernel(a32_tr, a32_te, cz8)
    scores["q32_qsvm"], _ = fit_predict_precomputed_svm(K32_tr, y_tr, p_tr, K32_te)

    pairs = [
        ("q8_fixed", "c7"), ("q8_fixed", "pca8"),
        ("q8_qsvm", "c7"), ("q8_qsvm", "rbf8"),
        ("q8_vqc_ent", "c7"), ("q8_vqc_ent", "q8_vqc_prod"),
        ("q8_zz_tuned", "c7"),
        ("q32_fixed", "c7"), ("q32_fixed", "pca32_log"),
        ("q32_qsvm", "c7"), ("q32_qsvm", "pca32_rbf"),
        ("pca32_log", "c7"),
    ]
    boot = patient_bootstrap_metrics(y_te, p_te, scores, pairs, n_draws=2000, seed=SEED)
    return {
        "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
        "n_test_patients": int(len(np.unique(p_te))),
        "n_test_positive": int(y_te.sum()),
        "test_prevalence": float(y_te.mean()),
        "boot": boot,
    }


def main():
    t0 = time.time()
    D = load_data()
    cz8, zs8 = _cz_ring_phase(8), _z_signs(8)
    m_poly = (D["roi_src"] == "lesion_polygon")

    print("1/3 Running PCA-32 / 32-qubit block-encoded 5-fold CV on full corpus (N=2,436, 328 pts) ...")
    cv32_full = run_pca32_cv(D["X_c7"], D["X_c6"], D["y"], D["pids"], cz8, zs8, "SMART_OM_full_2436")

    print("2/3 Running PCA-32 / 32-qubit block-encoded 5-fold CV on polygon subset (N=316, 160 pts) ...")
    cv32_poly = run_pca32_cv(D["X_c7"][m_poly], D["X_c6"][m_poly], D["y"][m_poly], D["pids"][m_poly], cz8, zs8, "A_lesion_polygon_316")

    print("3/3 Running FULL-CORPUS held-out test (N_test=381, 48 pts) ...")
    full_test = run_full_corpus_test(D, cz8, zs8)

    out = {"runtime_seconds": round(time.time() - t0, 2),
           "pca32_cv_2436": cv32_full, "pca32_cv_316": cv32_poly, "full_corpus_test_381": full_test}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE in {out['runtime_seconds']}s")

    for name, b in [("PCA-32 CV (N=2436)", cv32_full["boot"]),
                    ("PCA-32 CV (N=316)", cv32_poly["boot"]),
                    ("FULL TEST (N=381, 48 pts)", full_test["boot"])]:
        print(f"\n=== {name} ===")
        for k, v in b["point"].items():
            ci = b["ci_pr"][k]
            print(f"  {k:14s} PR={v['pr_auc']:.4f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  ROC={v['roc_auc']:.4f}")
        print("  Paired PR-AUC deltas:")
        for k, v in b["paired_pr"].items():
            print(f"    {k:28s} d={v['delta']:+.4f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
