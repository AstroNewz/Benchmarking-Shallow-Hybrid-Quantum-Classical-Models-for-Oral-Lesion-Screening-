#!/usr/bin/env python3
"""
Fast supplemental run completing the 4x8-qubit (32-D) block-encoded suite:
- Q32 VQC (L=2 non-commuting, 4 disjoint 8-qubit blocks -> 32 local <Z_j> -> linear head)
- Q32 VQC True Product State (zero CZ gates)
- Q32 Havlicek ZZ Tuned (gamma=0.1, 4 disjoint 8-qubit blocks -> 4x36=144 observables -> logistic)

Evaluated on:
1. 5-fold patient-grouped CV on full SMART-OM corpus (N=2,436, 328 pts)
2. 5-fold patient-grouped CV on A_lesion_polygon (N=316, 160 pts)
3. Full-corpus held-out test (N=381, 48 pts)

Outputs: paper/q32_vqc_zz.json
"""

import json
import math
import os
import sys
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paper.run_revision_experiments import (
    SEED,
    _apply_ry_torch,
    _cz_ring_phase,
    _havlicek_zz_states,
    _ry_product_states,
    _z_signs,
    _zz_observables,
    fit_predict_logistic,
    load_data,
    patient_bootstrap_metrics,
)
from paper.run_pca32_and_fulltest import MobileNetPCA32

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q32_vqc_zz.json")


def block32_zz_features(ang32, zs8, gamma=0.1):
    blocks = [_zz_observables(_havlicek_zz_states(ang32[:, b * 8:(b + 1) * 8] * gamma, reps=2), zs8) for b in range(4)]
    return np.hstack(blocks)


def train_and_score_vqc32_block(ang32_tr, y_tr, ang32_ev, *, entangle: bool, seed: int = SEED, steps: int = 80):
    """4 disjoint 8-qubit L=2 VQC blocks (64 circuit params + 33 head params = 97 trainable params).
    Each block b in {0..3} runs RY(alpha_b)->RY(th1_b)->[CZ_ring]->RY(th2_b)-><Z_j> (no cross-block entanglement)."""
    torch.manual_seed(seed)
    dt = torch.float32
    cz_t = torch.tensor(_cz_ring_phase(8), dtype=dt)
    zs_t = torch.tensor(_z_signs(8), dtype=dt)

    s_tr = [torch.tensor(_ry_product_states(ang32_tr[:, b * 8:(b + 1) * 8]), dtype=dt) for b in range(4)]
    s_ev = [torch.tensor(_ry_product_states(ang32_ev[:, b * 8:(b + 1) * 8]), dtype=dt) for b in range(4)]
    y_t = torch.tensor(y_tr, dtype=dt)
    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    w_t = torch.where(y_t == 1, len(y_tr) / (2.0 * max(n_pos, 1.0)), len(y_tr) / (2.0 * max(n_neg, 1.0)))

    gen = torch.Generator().manual_seed(seed)
    th1 = ((torch.rand(4, 8, generator=gen, dtype=dt) * 2.0 - 1.0) * math.pi).requires_grad_(True)
    th2 = ((torch.rand(4, 8, generator=gen, dtype=dt) * 2.0 - 1.0) * math.pi).requires_grad_(True)
    hw = (torch.randn(32, generator=gen, dtype=dt) * 0.1).requires_grad_(True)
    hb = torch.zeros((), dtype=dt, requires_grad=True)

    def forward(s_list):
        B = s_list[0].shape[0]
        shape = (B,) + (2,) * 8
        z_blocks = []
        for b in range(4):
            s = _apply_ry_torch(s_list[b].reshape(shape), th1[b], 8).reshape(B, 256)
            if entangle:
                s = s * cz_t
            s = _apply_ry_torch(s.reshape(shape), th2[b], 8).reshape(B, 256)
            z_blocks.append((s * s) @ zs_t.t())
        z_all = torch.cat(z_blocks, dim=1)
        return z_all @ hw + hb

    opt = torch.optim.Adam([th1, th2, hw, hb], lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        logits = forward(s_tr)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction="none") * w_t).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        return forward(s_ev).numpy()


def run_cv(X_c7, X_c6, y, pids, zs8, label):
    sgkf = StratifiedGroupKFold(n_splits=5)
    arms = ["c7", "pca32_log", "q32_vqc_ent", "q32_vqc_prod", "q32_zz_tuned"]
    oof = {a: np.zeros(len(y)) for a in arms}
    steps = 50 if len(y) > 500 else 100
    for fold, (ti, vi) in enumerate(sgkf.split(X_c7, y, pids)):
        sc7 = StandardScaler().fit(X_c7[ti])
        oof["c7"][vi] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            sc7.transform(X_c7[ti]), y[ti]).decision_function(sc7.transform(X_c7[vi]))

        pre = MobileNetPCA32().fit(X_c6[ti])
        p_tr, p_vi = pre.transform_pca(X_c6[ti]), pre.transform_pca(X_c6[vi])
        a_tr, a_vi = pre.transform_angles(X_c6[ti]), pre.transform_angles(X_c6[vi])

        scp = StandardScaler().fit(p_tr)
        oof["pca32_log"][vi] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scp.transform(p_tr), y[ti]).decision_function(scp.transform(p_vi))

        oof["q32_vqc_ent"][vi] = train_and_score_vqc32_block(a_tr, y[ti], a_vi, entangle=True, seed=SEED + fold, steps=steps)
        oof["q32_vqc_prod"][vi] = train_and_score_vqc32_block(a_tr, y[ti], a_vi, entangle=False, seed=SEED + fold, steps=steps)

        zz_tr = block32_zz_features(a_tr, zs8, gamma=0.1)
        zz_vi = block32_zz_features(a_vi, zs8, gamma=0.1)
        scz = StandardScaler().fit(zz_tr)
        oof["q32_zz_tuned"][vi] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
            scz.transform(zz_tr), y[ti]).decision_function(scz.transform(zz_vi))

    pairs = [("q32_vqc_ent", "c7"), ("q32_vqc_ent", "q32_vqc_prod"), ("q32_vqc_ent", "pca32_log"),
             ("q32_zz_tuned", "c7"), ("q32_zz_tuned", "pca32_log")]
    return patient_bootstrap_metrics(y, pids, oof, pairs, n_draws=2000, seed=SEED)


def main():
    t0 = time.time()
    D = load_data()
    zs8 = _z_signs(8)
    m_poly = (D["roi_src"] == "lesion_polygon")
    m_tr = (D["parts"] == "train")
    m_te = (D["parts"] == "test")

    print("1/3 32-D VQC & ZZ 5-fold CV on full corpus (N=2,436) ...")
    cv_full = run_cv(D["X_c7"], D["X_c6"], D["y"], D["pids"], zs8, "full_2436")

    print("2/3 32-D VQC & ZZ 5-fold CV on polygon subset (N=316) ...")
    cv_poly = run_cv(D["X_c7"][m_poly], D["X_c6"][m_poly], D["y"][m_poly], D["pids"][m_poly], zs8, "poly_316")

    print("3/3 32-D VQC & ZZ on full-corpus held-out test (N=381) ...")
    pre = MobileNetPCA32().fit(D["X_c6"][m_tr])
    a_tr, a_te = pre.transform_angles(D["X_c6"][m_tr]), pre.transform_angles(D["X_c6"][m_te])
    p_tr, p_te = pre.transform_pca(D["X_c6"][m_tr]), pre.transform_pca(D["X_c6"][m_te])
    scores = {}
    scores["c7"], _ = fit_predict_logistic(D["X_c7"][m_tr], D["y"][m_tr], D["pids"][m_tr], D["X_c7"][m_te])
    scores["pca32_log"], _ = fit_predict_logistic(p_tr, D["y"][m_tr], D["pids"][m_tr], p_te)
    scores["q32_vqc_ent"] = train_and_score_vqc32_block(a_tr, D["y"][m_tr], a_te, entangle=True, seed=SEED, steps=60)
    scores["q32_vqc_prod"] = train_and_score_vqc32_block(a_tr, D["y"][m_tr], a_te, entangle=False, seed=SEED, steps=60)
    scores["q32_zz_tuned"], _ = fit_predict_logistic(
        block32_zz_features(a_tr, zs8, gamma=0.1), D["y"][m_tr], D["pids"][m_tr],
        block32_zz_features(a_te, zs8, gamma=0.1)
    )
    pairs = [("q32_vqc_ent", "c7"), ("q32_vqc_ent", "q32_vqc_prod"), ("q32_zz_tuned", "c7")]
    te_full = patient_bootstrap_metrics(D["y"][m_te], D["pids"][m_te], scores, pairs, n_draws=2000, seed=SEED)

    out = {"runtime_seconds": round(time.time() - t0, 2), "cv_2436": cv_full, "cv_316": cv_poly, "test_381": te_full}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE in {out['runtime_seconds']}s")
    for title, b in [("CV 2436", cv_full), ("CV 316", cv_poly), ("TEST 381", te_full)]:
        print(f"\n=== {title} ===")
        for k, v in b["point"].items():
            ci = b["ci_pr"][k]
            print(f"  {k:14s} PR={v['pr_auc']:.4f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  ROC={v['roc_auc']:.4f}")
        for k, v in b["paired_pr"].items():
            print(f"    {k:28s} d={v['delta']:+.4f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
