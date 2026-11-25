# Benchmarking Shallow Hybrid Quantum–Classical Models for Oral Lesion Screening: A Controlled Empirical Study

**Authors:** Ishan Narayan Shukla\*, Pratyaksh Ranjan, Jay Karan Laxme, Rudransh Rajveer Singh, Khoirom Motilal Singh  
**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Information Technology Manipur, Imphal, India  
\*Corresponding author: `ishan256022@iiitmanipur.ac.in`

---

## Manuscript

- **Compiled IEEE double-column PDF:** [`paper/main.pdf`](paper/main.pdf)
- **LaTeX source & BibTeX:** [`paper/main.tex`](paper/main.tex), [`paper/references.bib`](paper/references.bib)
- **Publication figures (vector PDF & 300 DPI PNG):** [`paper/figures/`](paper/figures/)

## Key Findings

We evaluate shallow 8-qubit, globally entangled 16-qubit ($2^{16}$-D single-register), and $4 \times 8$-qubit block-encoded (32-D, disjoint-block) hybrid quantum–classical classifiers on the **SMART-OM** oral mucosa photography dataset (2,436 verified images across 328 patients, 5.9% prevalence; 316 clinician-annotated lesion-polygon images across 160 patients; strict $k = 0$ patient overlap) against matched-dimension classical controls (PCA-8/16/32 logistic and RBF-SVM, plus RFF and MLP at 8-D) and uncompressed classical references (C6 MobileNetV3 576-D; C7 Multimodal 1,195-D).

- **No evidence of quantum advantage:** Across 5-fold patient-grouped CV and held-out testing (381 full-corpus and 49 polygon test images), C7 achieves `0.5776` OOF PR-AUC (`0.9350` ROC-AUC) on the full 328-patient corpus, outperforming all 8-qubit models by `0.26–0.38` PR-AUC and globally entangled 16-qubit models by `0.23–0.25` PR-AUC.
- **Matched-dimension comparisons:** Global 16-qubit QSVM (`0.3503`) vs.\ PCA-16 RBF-SVM (`0.3375`) yields $\Delta = +0.013$ (`[-0.015, +0.042]`); $4 \times 8$ block-encoded QSVM (`0.4366`) vs.\ PCA-32 RBF-SVM (`0.4440`) yields $\Delta = -0.008$ (`[-0.021, +0.009]`).
- **Non-commuting entanglement ablation:** Verified non-commuting $L=2$ VQCs ($\max|\Delta\langle Z\rangle| = 0.997$) show no statistically significant gain over true product-state VQCs at 8-D or 32-D.
- **Bandwidth diagnostics:** Train-CV bandwidth tuning ($\gamma^* = 0.1$) resolves the untuned $n=8$ Havlíček $ZZ$ near-identity Gram collapse (effective rank `213.7 → 41.7`), yet tuned $ZZ$ still trails classical baselines on full-corpus CV.

## Reproducing All Experiments & Figures

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Exact pinned versions are recorded in `requirements.lock.txt`.)

2. **Re-run all benchmark suites** (uses the cached patient-disjoint features in `backend/artifacts/dataset/` and split manifest `backend/artifacts/dataset/split_manifest.json`):
   ```bash
   python paper/run_revision_experiments.py   # 8-qubit non-commuting suite, ZZ bandwidth sweep, paired bootstrap & 5-fold CV
   python paper/run_q16_global.py             # Globally entangled 16-qubit (2^16-D single-register) CV & held-out tests
   python paper/run_pca32_and_fulltest.py     # 4x8 block-encoded (32-D) Fixed/QSVM & 381-image full-corpus held-out test
   python paper/run_q32_vqc_zz.py             # 4x8 block-encoded (32-D) L=2 VQC & tuned ZZ
   ```

3. **Regenerate all 8 publication figures:**
   ```bash
   python paper/generate_figures.py
   ```

4. **Compile the IEEE LaTeX manuscript:**
   ```bash
   cd paper
   pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
   # or: tectonic -X compile main.tex
   ```

## Dataset Attribution

SMART-OM: *A SMARTphone based expert annotated dataset of Oral Mucosa images* (Figshare, posted February 18, 2026), authored by Dr.\ Madan Kumar P.\ D., Dr.\ Ranganathan K., Dr.\ Rajeshwari S., Dr.\ M.\ Pavithra, Dr.\ Lavanya C., Anwesh Nayak, Ramesh Kestur, Sushree Behera, and Raghuram Bharadwaj.
