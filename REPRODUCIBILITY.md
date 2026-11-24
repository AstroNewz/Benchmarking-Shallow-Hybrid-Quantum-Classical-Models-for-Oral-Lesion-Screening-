# REPRODUCIBILITY

How to reproduce the research results in this repository. Scope: the **algorithm and
experiments**, not the Flutter application or the web portal.

The authorised source for every number is
[`report/PAPER_MASTER_EVIDENCE_SHEET.md`](report/PAPER_MASTER_EVIDENCE_SHEET.md).
Numbers quoted in `README.md`, `report/main.tex`, `report.md`, `references/02`–`08` and
`references/index.html` are **not** usable — those files carry a
`FABRICATED-DATA WARNING` banner and the reason is recorded in
[`backend/artifacts/_QUARANTINE_FABRICATED/README.md`](backend/artifacts/_QUARANTINE_FABRICATED/README.md).

---

## 1. Environment

`requirements.lock.txt` is a full `pip freeze` of the interpreter that produced every
artifact in `backend/artifacts/`. `requirements.txt` specifies `>=` ranges and will not
reproduce the numbers exactly; use the lockfile.

```bash
python -m venv .venv && . .venv/Scripts/activate   # POSIX: . .venv/bin/activate
pip install -r requirements.lock.txt
```

The recorded environment is CPython **3.12.10**, win32 10.0.26200, with
`numpy==1.26.4`, `scikit-learn==1.9.0`, `scipy==1.15.3`, `scikit-image==0.25.2`,
`torch==2.13.0`, `torchvision==0.28.0`, `qiskit==1.4.6`, `qiskit-aer==0.17.2`.
Results are CPU-only; no GPU path is exercised by any experiment.

Every quantum number in this repository is an Aer **simulation**. See §6. Two details of
that claim were previously stated too strongly here and are corrected:

* `qiskit-ibm-runtime==0.40.1` **is** in `requirements.lock.txt:231`, so it is present in
  the environment that produced every artifact. `requirements.txt` does not declare it,
  which is why an earlier revision of this file called it "not a dependency" — that was
  wrong about the lockfile. It matters now because E8's device-calibrated noise model is
  built from `qiskit_ibm_runtime.fake_provider` and silently degrades to a synthetic
  depolarising model without it (§6).
* Hardware client code **does** exist: `quantum_ml/backends.py:239` `_connect_hardware`
  (imports `QiskitRuntimeService`) and `:456` `_expectation_hardware` (imports
  `SamplerV2`). It has never executed. `IBMQ_API_KEY` defaults to `None`
  (`backend/core/config.py:82`) and `_connect_hardware` returns
  `"no IBM Quantum API key configured"` before any network call is attempted. The honest
  statement is *reachable but never exercised*, not *absent*.

---

## 2. Dataset: SMART-OM (oral images)

### Attribution

`references/01_DATASETS.md:164` attributes the collection to the **Department of Oral
Medicine, Faculty of Dental Sciences, University of Peradeniya, Sri Lanka** (SMART-OM
Collaborative Medical Imaging Initiative).

### Acquisition — read this before trying

**No DOI, journal citation, licence file, or independent public mirror for SMART-OM is
recorded anywhere in this repository.** The only access path the repository names
(`references/01_DATASETS.md:165`) is a directory inside this project's own GitHub
repository, i.e. the dataset is **vendored, not independently published**. The copy used
for every result is the `SMART-OM/` directory at the repository root.

Consequences, stated plainly because they bound what reproduction can mean:

* A third party without this repository's `SMART-OM/` directory **cannot** reproduce the
  oral-track results, and this document does not pretend otherwise.
* There is no recorded licence, so redistribution terms are unknown.
* Any reuse should go through the attributed department. This file does not supply a
  contact, because the repository does not contain one.

`VERIFY WITH DATA OWNER`: licence, redistribution terms and a citable release.

### Expected layout

`backend/dataset/smartom.py` resolves directory *roles* by numeric prefix, so the names
below are matched case-insensitively and anything unmatched is reported rather than
dropped:

```
<root>/
  01. Normal/            01. Unannotated/<site>/<image>
                         02. Region annotation/...
                         03. Full annotation/...
  02. Variation from normal/   (plus 04. Lesion annotation/)
  03. OPMD/
  04. Oral Cancer/
  Descriptors/*.xlsx
  Metadata/Patient_s Metadata.xlsx
```

The canonical sample set is the **`01. Unannotated`** tree only. The other annotation
trees are re-renderings of the *same* source images with boxes and masks burned in;
counting them would duplicate every observation two to four times (DEC-015).

### Pointing the code at it

Autodiscovery (`discover_dataset_root`) checks `CARESCAN_DATASET_ROOT` first, then the
repository root, then its immediate subdirectories, for a folder carrying the numeric
class prefixes. To be explicit:

```bash
export CARESCAN_DATASET_ROOT=/absolute/path/to/SMART-OM
```

`.env.example` does not list this key; the environment variable above is the documented
mechanism, and `Settings.DATASET_ROOT` (`backend/core/config.py:93`) is the programmatic
equivalent.

### Verifying you have the same data

```bash
python -c "from backend.dataset.smartom import build_index; from backend.dataset.split import fingerprint_records; print(fingerprint_records(build_index().records))"
```

Expected: **`793cc12397ce3789`**, as recorded in
`backend/artifacts/dataset/split_manifest.json`. Every downstream phase quotes this
fingerprint; if it differs, nothing below will reproduce and the difference must be
resolved before continuing.

An earlier revision of this section printed `build_index().fingerprint`, which raises
`AttributeError` — `SmartOmIndex` has no such attribute, and the digest is computed by
`fingerprint_records` in `backend/dataset/split.py:312` over `(image_id, patient_id,
diagnostic_class)` triples in `image_id` order. The command above is verified to print the
expected value (ISS-030). This mattered more than a typo: it was the first verification
step in the document, so a third party following it hit an exception before reaching
anything else.

Derived counts that must also hold: 2,469 source images → 33 excluded by quality control
→ **2,436** kept, over **328** patients, split 0.70 / 0.15 / 0.15 at patient level with
`seed = 42` into 1,692 / 363 / 381 images and 104 / 21 / 18 positives, where positive is
`{opmd, oral_cancer}`.

---

## 3. Dataset: PTB-XL (ECG, the E4 track only)

Public and properly citable, unlike the above:

* Wagner, P., Strodthoff, N., Bousseljot, R. D., et al. "PTB-XL, a large publicly
  available electrocardiography dataset." *Scientific Data* **7**, 154 (2020).
  DOI [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)
* Version **1.0.3**, <https://physionet.org/content/ptb-xl/1.0.3/>
* `backend/dataset/ptbxl.py` verifies the download against the release's own
  `SHA256SUMS.txt` via `verify_checksums`.

PTB-XL is a **separate dataset and a separate modality**, used because the oral set was
too small to answer E4's question. No PTB-XL result transfers to the oral algorithm.

---

## 4. Reproducing the experiments

Each module writes one JSON artifact and reads frozen inputs. Determinism comes from
`seed = 42` plus SHA-256-ordered patient allocation; repeat runs and cache reloads were
verified to produce identical artifacts.

| Phase | Command | Artifact |
|---|---|---|
| Split + QC | `python -m backend.dataset.split` | `backend/artifacts/dataset/split_manifest.json` |
| E0 controls / readout | `python -m backend.evaluation.e0_controls` | `reports_e0_*.json` |
| E1 failure map | `python -m backend.evaluation.e1_failure_map` | `reports_e1_failure_map.json` |
| E2 angle / visual | `python -m backend.evaluation.e2_angle_experiment` | `reports_e2_*.json` |
| E3 kernel / VQC / fusion | `python -m backend.evaluation.e3_quantum_kernel` (and siblings) | `reports_e3_*.json` |
| E4 ECG arena | `python -m backend.evaluation.e4_quantum` | `backend/artifacts/reports/e4_*.json` |
| E5 post-hoc ablation | `python -m backend.evaluation.e5_posthoc_ablation` | `backend/artifacts/reports/reports_e5_posthoc_ablation.json` |
| E6 test-partition intervals | `python -m backend.evaluation.e6_test_intervals` | `backend/artifacts/reports/reports_e6_test_intervals.json` |
| E7 dimensionality ladder | `python -m backend.evaluation.e7_dimension_ladder` | `backend/artifacts/reports/reports_e7_dimension_ladder.json` |
| E8 noisy simulation | `python -m backend.evaluation.e8_noisy_simulation` | `backend/artifacts/reports/reports_e8_noisy_simulation.json` |
| E9 duplicate / near-duplicate scan | `python -m backend.evaluation.e9_duplicate_scan` | `backend/artifacts/reports/reports_e9_duplicate_scan.json` |
| Frozen test evaluation | `python -m backend.evaluation.pixel_final_test` | `reports_final_{oracle,predicted}.json` |

Measured wall-clock, same machine: E5 **2,439.4 s**; E6 **78.5 s** (after a ~25 s model
refit). The quantum phases are the expensive ones; E3's kernel arm is quadratic in rows.

Tests:

```bash
python -m pytest -q
```

**1,038** tests pass in the recorded environment — measured, `1038 passed in 363.93s`, with
`1038 tests collected` and nothing skipped, deselected or erroring.

**Not 1,231.** That figure appears throughout the repository and is a **Python-plus-Dart
sum**: `references/README.md:108` decomposes it as "1,038 backend in 732s + 193 mobile in
10s". `python -m pytest` cannot produce it, because 193 of those are Flutter tests run by
`flutter test`. An earlier revision of this section printed 1,231 directly beneath the
`pytest` command above, which was a category error (ISS-028). The measured wall-clock also
differs from the recorded 732 s; 363.93 s is what this machine does now, and neither number
is a count.

**What the count does and does not cover (ISS-027).** `tests/` holds a module per phase
up to E4 and nothing for **E5–E8**: those four are not imported by any test. Do not read
the figure above as validation of the last four rows of the table. Their guarantee is a
different one — each asserts frozen upstream references before it writes anything, and §5
names exactly which. The distinction matters because ISS-026 was a real defect inside E8
that the anchor gates could not see; it was found by transpiling independently and diffing.

---

## 5. Which numbers are self-checking

Several artifacts refuse to be written if they cannot reproduce a reference value. These
are the cheapest reproduction checks available, because a mismatch halts the run instead
of producing a plausible wrong number:

* **E5** reproduces E1 exactly where the two overlap: `L_H.random_forest = 0.884779`
  (E1's C1) and `L_CHM.logistic = 0.913038` (E1's C7), to six decimals.
* **E6** re-derives all 18 frozen test-partition point estimates and all 5 leak bars
  before attaching an interval to any of them, and raises `IntervalError` on a mismatch.
  `random_forest` on `B_localized` reproduces PR-AUC `0.592555906676057` — the clean
  headline — and `A_all` reproduces ROC-AUC `0.9410774410774411`.
* **E7**'s unreduced MobileNet rung reproduces E5's `L_C` cell (`0.892227`), which in
  turn reproduces E1's C6.
* **E8** asserts four E3 reference metrics to **1e-6** before running a single noisy
  circuit — `c7 = 0.913038`, `quantum = 0.868335`, `pca = 0.838846`, `rff = 0.795360` —
  and aborts the phase on any mismatch. Its own rung values were additionally verified
  reproducible bit-for-bit across repeat runs (all six: 0.877043 / 0.886123 / 0.889416
  noiseless, 0.862819 / 0.892494 / 0.878441 noisy).
* E1 carries an anchor assertion that halts the phase if a reference metric drifts.
* **E9** refuses to report any count unless `fingerprint_records(build_index().records)`
  equals `793cc12397ce3789`, so a duplicate scan can never be attributed to a dataset it
  did not run on. It also re-checks every hash-flagged pair for pixel identity, so exact
  and approximate matches cannot be conflated in the artifact.

Measured wall-clock for the two newest phases, same machine: E8 **5,168.4 s** complete
(of which the pre-registered rungs are 420.4 s; the post-hoc routed arm dominates), E9
**75.9 s**.

---

## 6. What cannot be reproduced, and why

* **Hardware execution.** None happened. The artifacts that claimed it are quarantined
  and their generator is kept as the proof. The genuine hardware result is an
  *infeasibility bound* from measured transpilations at 6/8/10/12 qubits, extrapolated
  to 16 (`measured: false` at 16 qubits).
* **The oral dataset, by a third party.** See §2.
* **A noisy-simulator arm.** Superseded: E8 now exercises it. `fake_sherbrooke` supplies a
  **device-calibrated** `NoiseModel` (`noise.device_calibrated_noise: true`, 127 noisy
  qubits), and `reports_e8_noisy_simulation.json` contains a three-rung
  ideal/shots/noisy ladder. Two limits on what that buys, both recorded in the artifact
  itself: it is a **noise model built from published calibration data, not an execution**
  — nothing ran on a QPU, and E8 cannot substitute for hardware; and the pre-registered
  rungs transpile to the device **basis only**, with the coupling map imposed in a
  separate post-hoc `layout_sensitivity` arm (ISS-026). Without
  `qiskit-ibm-runtime` installed the arm silently degrades to a synthetic depolarising
  model and E8 sets `noise.claim_withheld` rather than reporting a device result
  (ISS-025).
* **Anything from the quarantine directory.** Its generator is seeded, so re-running it
  reproduces the fabricated numbers exactly. **Reproducing a fabrication is not
  verification.** Do not run it to "check" anything.
