#!/usr/bin/env python3
"""
Generate publication figures for:
"Evaluating Hybrid Quantum-Classical Classifiers for Oral Lesion Screening:
 A Systematic Negative-Result Study"

All figures are rendered at their final IEEE print size (single column = 3.5 in,
double column = 7.16 in) so LaTeX includes them at 1.0x scale with crisp 6-7.5 pt text.
All numbers trace directly to verified repository JSON reports and paper/revision_metrics.json.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL_W = 3.5    # IEEE single-column width (in)
TXT_W = 7.16   # IEEE double-column text width (in)

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 7,
    'axes.labelsize': 7.2,
    'axes.titlesize': 7.3,
    'xtick.labelsize': 6.3,
    'ytick.labelsize': 6.3,
    'legend.fontsize': 6.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
})

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)


def save_fig(fig, base_name):
    for folder in [PAPER_DIR, FIGS_DIR]:
        fig.savefig(os.path.join(folder, f"{base_name}.pdf"))
        fig.savefig(os.path.join(folder, f"{base_name}.png"), dpi=300)
    print(f"  [SAVED] {base_name} ({fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in)")
    plt.close(fig)


def _box(ax, x, y, w, h, title, body, bg, border, tsize=6.5, bsize=5.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                fc=bg, ec=border, lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h - 0.05, title, ha='center', va='top', fontsize=tsize,
            fontweight='bold', color=border, zorder=3)
    if body:
        ax.text(x + w / 2, y + (h - 0.16) / 2, body, ha='center', va='center',
                fontsize=bsize, color='#212121', linespacing=1.18, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#37474f", lw=0.9, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                mutation_scale=7, shrinkA=0, shrinkB=0), zorder=1)


# ==============================================================================
# Figure 1: System Pipeline Architecture (double column)
# ==============================================================================
def fig1_pipeline_architecture():
    W, H = TXT_W, 2.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    c_data, b_data = '#e8f4f8', '#0277bd'
    c_cls, b_cls = '#e8f5e9', '#2e7d32'
    c_q, b_q = '#f3e5f5', '#7b1fa2'
    c_ev, b_ev = '#fff8e1', '#e65100'

    _box(ax, 0.05, 1.62, 1.02, 0.90, "SMART-OM corpus", "2,436 imgs / 328 pts\n316 lesion polygons\n(160 patients)", c_data, b_data)
    _box(ax, 0.05, 0.38, 1.02, 0.90, "Lesion ROI", "Polygonal crop\n(A_lesion_polygon)\n& ROI leak bar", c_data, b_data)
    _arrow(ax, 0.56, 1.62, 0.56, 1.28)

    # C6 feeds BOTH C7 fusion AND PCA-8 for quantum circuits
    _box(ax, 1.32, 1.65, 1.14, 0.82, "C6: MobileNetV3", "Frozen ImageNet\nembedding, 576-D", c_cls, b_cls)
    _box(ax, 1.32, 0.42, 1.14, 0.82, "C5: Morphology", "CIELAB + GLCM,\n8×8 grid, 619-D", c_cls, b_cls)
    _arrow(ax, 1.07, 0.83, 1.32, 2.02)
    _arrow(ax, 1.07, 0.83, 1.32, 0.83)

    _box(ax, 2.68, 0.42, 1.05, 0.82, "C7: Fusion", "hstack[C6, C5]\n1,195-D classical\nreference", c_cls, b_cls)
    _arrow(ax, 2.46, 1.85, 2.68, 1.00)
    _arrow(ax, 2.46, 0.83, 2.68, 0.83)

    # PCA-8 / PCA-16 / PCA-32 fed directly from C6 (MobileNetV3 576-D)
    _box(ax, 3.92, 1.56, 1.08, 0.96, "Train-only PCA", "C6: 576-D → 8/16/32\n49 / 64 / 77% var.\nα ∈ [0, π/2]", c_data, b_data)
    _arrow(ax, 2.46, 2.06, 3.92, 2.06)

    qx, qw, qh = 5.22, 1.08, 0.50
    q_specs = [
        (2.18, "Fixed hybrid", r"$R_y\!\to\!CZ\!\to\!R_y(\frac{\pi}{4})$", c_q, b_q),
        (1.58, "Fidelity QSVM", r"$|\langle\psi|\psi'\rangle|^2$ kernel", c_q, b_q),
        (0.98, "Trainable VQC", r"$L=2$ non-commuting", c_q, b_q),
        (0.38, "Havlíček ZZ map", r"Tuned $\gamma^*=0.1$", '#ffebee', '#c62828'),
    ]
    for y, t, b, bg, bd in q_specs:
        _box(ax, qx, y, qw, qh, t, b, bg, bd, tsize=6.3, bsize=5.6)
        _arrow(ax, 5.00, 2.04, qx, y + qh / 2)

    ex, ew = 6.50, 0.62
    _box(ax, ex, 0.80, ew, 1.32, "Evaluation", "Patient-\ndisjoint\nbootstrap\n& 5-fold\ngrouped CV", c_ev, b_ev,
         tsize=6.2, bsize=5.6)
    for y, *_ in q_specs:
        _arrow(ax, qx + qw, y + qh / 2, ex, 1.46)

    gy = 0.14
    ax.plot([3.20, 3.20, ex + ew / 2], [0.42, gy, gy], color=b_cls, lw=1.0, ls='--', zorder=1)
    _arrow(ax, ex + ew / 2, gy, ex + ew / 2, 0.80, color=b_cls, lw=1.0)
    ax.text(4.95, gy + 0.04, "Direct classical reference: C7 (1,195-D) & matched 8/16/32-D controls",
            ha='center', va='bottom', fontsize=5.8, fontweight='bold', color=b_cls)

    save_fig(fig, "fig1_pipeline_architecture")


# ==============================================================================
# Figure 2: Quantum Circuit Paradigms (double column, 2x2) - non-commuting
# ==============================================================================
def fig2_quantum_circuits():
    fig, axes = plt.subplots(2, 2, figsize=(TXT_W, 3.35))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.06, hspace=0.22)
    axes = axes.flatten()
    titles = [
        "(a) Fixed hybrid encoding (non-commuting readout)",
        "(b) Fidelity quantum kernel (QSVM)",
        "(c) Trainable L=2 VQC (non-commuting entangler)",
        "(d) Bandwidth-tuned Havlíček ZZ feature map",
    ]
    yq = [3.9, 2.75, 1.35]
    qlabels = [r"$|0\rangle_{q_0}$", r"$|0\rangle_{q_1}$", r"$|0\rangle_{q_7}$"]

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.6)
        ax.axis('off')
        ax.set_title(title, fontsize=7.1, fontweight='bold', pad=2)
        for y, lbl in zip(yq, qlabels):
            ax.plot([0.95, 9.7], [y, y], color='#37474f', lw=0.8, zorder=1)
            ax.text(0.85, y, lbl, va='center', ha='right', fontsize=6.5)
        ax.text(0.55, 2.05, r"$\vdots$", va='center', ha='center', fontsize=8)

        def gate(x, y, text, w=1.22, h=0.72, fc='#e1bee7', ec='#8e24aa'):
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=fc, ec=ec, lw=0.8, zorder=3))
            ax.text(x, y, text, ha='center', va='center', fontsize=6.1, zorder=4)

        def cz(x, y1, y2):
            ax.plot([x, x], [y1, y2], color='#00695c', lw=1.0, zorder=2)
            ax.plot([x, x], [y1, y2], 'o', color='#00695c', ms=3.0, zorder=4)

        def meter(x, y, txt=r"$\langle Z\rangle$"):
            ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.36), 0.84, 0.72, boxstyle="square,pad=0.0",
                                        fc='#eceff1', ec='#455a64', lw=0.8, zorder=3))
            ax.text(x, y, txt, ha='center', va='center', fontsize=6.0, zorder=4)

        def note(txt, color):
            ax.text(5.3, 0.36, txt, ha='center', va='center', fontsize=6.1, color=color, fontweight='bold')

        if idx == 0:
            for y, a in zip(yq, [r"$R_y(\alpha_0)$", r"$R_y(\alpha_1)$", r"$R_y(\alpha_7)$"]):
                gate(1.9, y, a, fc='#bbdefb', ec='#1976d2')
            cz(3.3, yq[0], yq[1]); cz(4.2, yq[1], yq[2]); cz(5.1, yq[0], yq[2])
            for y in yq:
                gate(6.5, y, r"$R_y(\frac{\pi}{4})$", w=1.18, fc='#c8e6c9', ec='#2e7d32')
                meter(8.4, y)
            note(r"16 observables ($\langle Z_j\rangle, \langle Z_j Z_{j+1}\rangle$); $R_y(\pi/4)$ breaks $CZ$–$Z$ commutation", '#0d47a1')
        elif idx == 1:
            for y in yq:
                gate(2.6, y, r"$U_{\mathrm{enc}}(x)$", w=1.7, fc='#b2dfdb', ec='#00796b')
                gate(5.2, y, r"$U^{\dagger}_{\mathrm{enc}}(x')$", w=1.7, fc='#ffe0b2', ec='#f57c00')
                meter(8.0, y, r"$P(0^n)$")
            note(r"Non-product $U_{\mathrm{enc}}=R_y(\pi/4)\,CZ\,R_y(\alpha)$; $k(x,x')=|\langle\psi|\psi'\rangle|^2$", '#004d40')
        elif idx == 2:
            for y, a in zip(yq, [r"$R_y(\alpha_0)$", r"$R_y(\alpha_1)$", r"$R_y(\alpha_7)$"]):
                gate(1.75, y, a, w=1.1, fc='#bbdefb', ec='#1976d2')
            for y, t in zip(yq, [r"$R_y(\theta_0^{(1)})$", r"$R_y(\theta_1^{(1)})$", r"$R_y(\theta_7^{(1)})$"]):
                gate(3.35, y, t, w=1.22, fc='#d1c4e9', ec='#512da8')
            cz(4.65, yq[0], yq[1]); cz(5.25, yq[1], yq[2])
            for y, t in zip(yq, [r"$R_y(\theta_0^{(2)})$", r"$R_y(\theta_1^{(2)})$", r"$R_y(\theta_7^{(2)})$"]):
                gate(6.65, y, t, w=1.22, fc='#d1c4e9', ec='#512da8')
            for y in yq:
                meter(8.5, y)
            note(r"$CZ$ between $R_y(\theta^{(1)})$ and $R_y(\theta^{(2)})$ ($\max|\Delta\langle Z\rangle|=0.997$); ablation drops $CZ$", '#4a148c')
        else:
            for y in yq:
                gate(1.75, y, r"$H$", w=0.65, fc='#fff9c4', ec='#f9a825')
                gate(3.35, y, r"$U_\Phi(\gamma x)$", w=1.45, fc='#ffcdd2', ec='#d32f2f')
                gate(4.95, y, r"$H$", w=0.65, fc='#fff9c4', ec='#f9a825')
                gate(6.55, y, r"$U_\Phi(\gamma x)$", w=1.45, fc='#ffcdd2', ec='#d32f2f')
                meter(8.5, y)
            note(r"36 observables ($\langle Z_i\rangle, \langle Z_i Z_j\rangle$); bandwidth $\gamma$ tuned on train CV ($\gamma^*=0.1$)", '#b71c1c')

    save_fig(fig, "fig2_quantum_circuits")


# ==============================================================================
# Figure 3: 5-Fold Patient-Grouped CV Leaderboard (complete parity with Table III)
# ==============================================================================
def fig3_leaderboard():
    arms = ["C7\n1195D", "C6\n576D", "PCA32\nRBF", "Q32\nFix", "Q32\nQSVM", "Q32\nVQC", "Q32\nZZ",
            "PCA8\nRBF", "Q8\nFix", "Q8\nQSVM", "Q8\nVQC", "Q8\nZZ"]
    colors = ['#2e7d32', '#1b5e20', '#546e7a', '#0277bd', '#00838f', '#6a1b9a', '#b71c1c',
              '#78909c', '#1976d2', '#0288d1', '#7b1fa2', '#e53935']

    # Panel (a): Full SMART-OM corpus (N=2,436, 328 patients, prev=0.059)
    pr_full = [0.5776, 0.4764, 0.4440, 0.4248, 0.4366, 0.4105, 0.2437, 0.3033, 0.2675, 0.3160, 0.2619, 0.1929]
    lo_full = [0.473, 0.379, 0.355, 0.344, 0.353, 0.329, 0.185, 0.217, 0.200, 0.238, 0.203, 0.144]
    hi_full = [0.688, 0.590, 0.553, 0.521, 0.545, 0.514, 0.326, 0.403, 0.353, 0.415, 0.343, 0.260]

    # Panel (b): A_lesion_polygon subset (N=316, 160 patients, prev=0.449)
    pr_poly = [0.7560, 0.7271, 0.7129, 0.7154, 0.7169, 0.6856, 0.6224, 0.6889, 0.6882, 0.6870, 0.6877, 0.6284]
    lo_poly = [0.664, 0.636, 0.622, 0.622, 0.625, 0.595, 0.533, 0.589, 0.593, 0.583, 0.587, 0.538]
    hi_poly = [0.840, 0.810, 0.802, 0.805, 0.806, 0.787, 0.729, 0.783, 0.787, 0.785, 0.788, 0.733]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.65), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.94, bottom=0.125, hspace=0.28)
    x = np.arange(len(arms))

    def draw(ax, pts, lo, hi, prev, title, ylim):
        err = [[p - l for p, l in zip(pts, lo)], [h - p for p, h in zip(pts, hi)]]
        ax.bar(x, pts, yerr=err, capsize=1.5, error_kw=dict(lw=0.6, capthick=0.6, ecolor='#263238'),
               color=colors, width=0.66, edgecolor='black', linewidth=0.45, zorder=3)
        ax.axhline(pts[0], color='#2e7d32', ls='--', lw=0.8, zorder=2)
        ax.axhline(prev, color='gray', ls=':', lw=0.8, zorder=2)
        ax.text(len(arms) - 0.55, prev + (ylim[1] - ylim[0]) * 0.02, f"prevalence ({prev:.3f})",
                ha='right', va='bottom', fontsize=5.2, color='#333333',
                bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.88), zorder=5)
        for i, (p, h) in enumerate(zip(pts, hi)):
            ax.text(i, h + (ylim[1] - ylim[0]) * 0.02, f"{p:.2f}", ha='center', va='bottom',
                    fontsize=4.7, fontweight='bold',
                    bbox=dict(fc='white', ec='none', pad=0.15, alpha=0.82), zorder=4)
        ax.set_ylim(*ylim)
        ax.set_ylabel("OOF PR-AUC")
        ax.set_title(title, fontsize=6.8, fontweight='bold', pad=2)
        ax.grid(axis='y', ls='--', alpha=0.5, zorder=0)

    draw(a1, pr_full, lo_full, hi_full, 0.059, "(a) Full SMART-OM corpus (N=2,436, 328 patients)", (0.0, 0.83))
    draw(a2, pr_poly, lo_poly, hi_poly, 0.449, "(b) Lesion-polygon subset (N=316, 160 patients)", (0.35, 0.95))
    a2.set_xticks(x)
    a2.set_xticklabels(arms, fontsize=4.9, linespacing=1.0)
    save_fig(fig, "fig3_leaderboard")


# ==============================================================================
# Figure 4: Dimensionality ladder (single column)
# ==============================================================================
def fig4_dimension_ladder():
    dims = [4, 8, 16, 32, 64, 128]
    pr_mob = [0.710344, 0.838846, 0.879133, 0.908709, 0.826190, 0.705052]
    pr_c7 = [0.8077, 0.7384, 0.6948, 0.7325, 0.6915, 0.6938]
    var = [38.2, 49.1, 63.8, 76.9, 85.3, 93.1]

    fig, ax1 = plt.subplots(figsize=(COL_W, 2.45))
    fig.subplots_adjust(left=0.135, right=0.865, top=0.80, bottom=0.15)
    c1, c7c, c2 = '#1565c0', '#6a1b9a', '#e65100'
    ax1.axvspan(3.4, 9.4, color='#ffebee', zorder=0)
    l1 = ax1.plot(dims, pr_mob, marker='s', ms=3.5, color=c1, lw=1.3, label='MobileNetV3 PCA ladder')
    l2 = ax1.plot(dims, pr_c7, marker='^', ms=3.5, color=c7c, lw=1.0, ls='-.', label='C7 PCA ladder')
    l3 = ax1.axhline(0.913038, color='#2e7d32', ls='--', lw=0.9, label='Unreduced C7 (0.913)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(dims)
    ax1.set_xticklabels([str(d) for d in dims])
    ax1.minorticks_off()
    ax1.set_xlim(3.2, 150)
    ax1.set_ylim(0.65, 0.95)
    ax1.set_xlabel("PCA dimension $d$ (design qubit budget $n=d$)")
    ax1.set_ylabel("Validation PR-AUC", color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.text(5.75, 0.947, "≤8-qubit\ndesign", ha='center', va='top', fontsize=5.3, color='#c62828')

    ax2 = ax1.twinx()
    l4 = ax2.plot(dims, var, marker='o', ms=3, color=c2, lw=1.0, ls=':', label='Cum. PCA variance (%)')
    ax2.set_ylim(30, 100)
    ax2.set_ylabel("Cum. PCA variance (%)", color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)

    ax1.annotate("d = 32: 0.909 (76.9% var.)", xy=(31, 0.912), xytext=(10.5, 0.924),
                 fontsize=5.5, ha='left', arrowprops=dict(arrowstyle='-|>', lw=0.6, mutation_scale=6))
    ax1.annotate("d = 8: 0.839\n(49.1% var.)", xy=(7.6, 0.842), xytext=(3.5, 0.858),
                 fontsize=5.5, ha='left', arrowprops=dict(arrowstyle='-|>', lw=0.6, mutation_scale=6))

    lines = l1 + l2 + [l3] + l4
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, columnspacing=1.2, handlelength=2.0)
    save_fig(fig, "fig4_dim_ladder")


# ==============================================================================
# Figure 5: Noise robustness ladder (single column)
# ==============================================================================
def fig5_noise_robustness():
    rungs = ["Rung 1\nIdeal statevector", "Rung 2\n1,024 shots", "Rung 3\nFakeSherbrooke\n+ 1,024 shots"]
    margins = [0.029489, 0.045348, 0.039072]
    errs = [0.0, 0.0, 0.014844]

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.97, bottom=0.21)
    x = np.arange(3)
    ax.bar(x, margins, yerr=errs, capsize=3, error_kw=dict(lw=0.8, capthick=0.8),
           color=['#90caf9', '#42a5f5', '#1e88e5'], edgecolor='black', linewidth=0.5, width=0.5, zorder=3)
    for i, (m, e) in enumerate(zip(margins, errs)):
        lbl = f"+{m:.4f}" + (f"\n± {e:.4f}" if e else "")
        ax.text(i + (0.29 if e else 0), m + (0.002 if not e else -0.006), lbl,
                ha='left' if e else 'center', va='bottom', fontsize=6.0, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rungs, fontsize=6.2, linespacing=1.05)
    ax.set_ylabel("PR-AUC margin vs. PCA-8")
    ax.set_ylim(0, 0.086)
    ax.set_xlim(-0.5, 3.0)
    ax.grid(axis='y', ls='--', alpha=0.5, zorder=0)
    ax.text(1.22, 0.0815, "Margin vs. PCA-8 survives noise (+0.0391 ± 0.0148)\n"
                          "Increment over C7 fails gating null (p = 0.095)",
            ha='center', va='top', fontsize=5.8, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff9c4", ec="#f9a825", lw=0.7))
    save_fig(fig, "fig5_noise_robustness")


# ==============================================================================
# Figure 6: Held-out test partition forest plot (C7 + Quantum + Controls + Leak Bar)
# ==============================================================================
def fig6_test_intervals():
    models = [
        "C7 (1,195-D)", "C6 (MobileNetV3)", "PCA-8 control", "RBF-SVM ctrl",
        "Fixed Q (nc)", "Fidelity QSVM", "VQC L=2 (ent.)", "VQC L=2 (prod.)",
        "ZZ tuned (γ=0.1)", "ZZ untuned (γ=1)", "ROI leak bar"
    ]
    pr = [0.7054, 0.7364, 0.6005, 0.6520, 0.5783, 0.6601, 0.6223, 0.6263, 0.7226, 0.4165, 0.3417]
    pr_lo = [0.459, 0.488, 0.359, 0.395, 0.345, 0.416, 0.346, 0.380, 0.478, 0.226, 0.196]
    pr_hi = [0.933, 0.926, 0.807, 0.871, 0.809, 0.869, 0.853, 0.828, 0.895, 0.649, 0.564]
    colors = ['#2e7d32', '#2e7d32', '#607d8b', '#607d8b', '#1565c0', '#1565c0', '#7b1fa2', '#9c27b0', '#e65100', '#c62828', '#424242']

    y = np.arange(len(models))[::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    fig.subplots_adjust(left=0.32, right=0.975, top=0.95, bottom=0.13)

    for yi, p, l, h, c in zip(y, pr, pr_lo, pr_hi, colors):
        ax.errorbar(p, yi, xerr=[[p - l], [h - p]], fmt='o', color=c, ecolor=c,
                    elinewidth=1.0, capsize=2.0, capthick=0.7, ms=3.2, zorder=3)
    ax.axvline(0.367, color='gray', ls='--', lw=0.8, zorder=1)
    ax.text(0.38, len(models) - 0.35, "prevalence (0.367)", fontsize=5.5, color='#555555', va='center')
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=6.0)
    ax.set_xlim(0.15, 1.00)
    ax.set_ylim(-0.6, len(models) - 0.05)
    ax.set_xlabel("Held-out test PR-AUC [95% patient bootstrap CI]")
    ax.grid(True, ls='--', alpha=0.5, zorder=0)
    save_fig(fig, "fig6_test_intervals")


# ==============================================================================
# Figure 7: Classical representation hierarchy (single column)
# ==============================================================================
def fig7_classical_ablation():
    cands = ["C1\n163", "C2\n1,024", "C2b\n256", "C3\n144", "C4\n96", "C5\n619", "C6\n576", "C7\n1,195"]
    pr = [0.8848, 0.6158, 0.6091, 0.6633, 0.7381, 0.8558, 0.8894, 0.9130]
    colors = ['#90a4ae'] * 7 + ['#2e7d32']
    colors[5] = colors[6] = '#1976d2'

    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.97, bottom=0.235)
    x = np.arange(len(cands))
    bars = ax.bar(x, pr, color=colors, width=0.64, edgecolor='black', linewidth=0.5, zorder=3)
    for b, s in zip(bars, pr):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.008, f"{s:.3f}", ha='center', va='bottom',
                fontsize=5.6, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cands, fontsize=6.2, linespacing=1.0)
    ax.set_xlabel("Candidate representation (dimension $D$)", labelpad=1.5)
    ax.set_ylabel("Validation PR-AUC")
    ax.set_ylim(0.5, 0.97)
    ax.grid(axis='y', ls='--', alpha=0.5, zorder=0)
    save_fig(fig, "fig7_classical_ablation")


# ==============================================================================
# Figure 8: Patient cohort funnel (single column) - EXACT patient/image counts
# ==============================================================================
def fig8_patient_funnel():
    W, H = COL_W, 3.20
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    _box(ax, 0.20, 2.62, 2.15, 0.52, "Raw SMART-OM repository",
         "2,469 images · 328 patients", '#e3f2fd', '#1565c0')
    _box(ax, 2.48, 1.80, 0.98, 0.78, "QC exclusions",
         "33 images removed\n24 motion blur\n3 out of focus\n6 overexposed", '#ffebee', '#c62828',
         tsize=6.1, bsize=5.4)
    _box(ax, 0.20, 1.62, 2.15, 0.62, "Verified full corpus (5-fold CV)",
         "2,436 images · 328 patients\n143 pos. imgs (5.9%) · 81 pos. pts", '#e8f5e9', '#2e7d32')
    _arrow(ax, 1.27, 2.62, 1.27, 2.24)
    _arrow(ax, 1.27, 2.42, 2.48, 2.42)

    _box(ax, 0.20, 1.00, 3.10, 0.44, "Primary lesion-polygon subset (A_lesion_polygon)",
         "316 clinician-annotated images · 160 patients (k = 0 overlap)", '#ede7f6', '#5e35b1',
         tsize=6.2, bsize=5.6)
    _arrow(ax, 1.27, 1.62, 1.27, 1.44)

    parts = [
        (0.04, "Train split", "Full: 1,692 imgs (230 pts)\nPoly: 215 imgs (112 pts)\nfit scaler & PCA", '#f3e5f5', '#7b1fa2'),
        (1.20, "Validation split", "Full: 363 imgs (50 pts)\nPoly: 52 imgs (25 pts)\nselection & E8", '#e0f7fa', '#00838f'),
        (2.36, "Held-out test", "Full: 381 imgs (48 pts)\nPoly: 49 imgs (23 pts)\nlocked, run once", '#fff8e1', '#e65100'),
    ]
    for (px, t, b, bg, bd), sx in zip(parts, [0.75, 1.75, 2.75]):
        _box(ax, px, 0.05, 1.10, 0.80, t, b, bg, bd, tsize=5.8, bsize=5.1)
        _arrow(ax, sx, 1.00, px + 0.55, 0.85)

    save_fig(fig, "fig8_patient_funnel")


if __name__ == '__main__':
    print("Rendering revised figures at IEEE print size ...")
    fig1_pipeline_architecture()
    fig2_quantum_circuits()
    fig3_leaderboard()
    fig4_dimension_ladder()
    fig5_noise_robustness()
    fig6_test_intervals()
    fig7_classical_ablation()
    fig8_patient_funnel()
    print("Done.")
