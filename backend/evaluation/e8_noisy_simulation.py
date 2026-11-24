"""E8 -- Does the E2/E3 quantum readout survive device noise, or is it noiseless-only?

Phase E8 closes the one remaining gap the paper audit
(``report/PAPER_MASTER_EVIDENCE_SHEET.md``) lists as closable: ``QUANTUM_NOISE_MODEL =
"fake_sherbrooke"`` has been configured since the first commit of
``backend/core/config.py`` and has **never been exercised**. Every quantum number in
this repository was produced by *exact, noiseless* statevector simulation. No circuit
has ever run on hardware (see ``backend/artifacts/_QUARANTINE_FABRICATED/README.md``;
the receipts that claimed otherwise are quarantined fabrications and are not evidence).

This phase does **not** substitute for hardware and does not pretend to. It measures
the next best thing that is honestly available: the same circuit, the same rows, the
same downstream head, evaluated through an Aer noise model built from IBM's **published
calibration data** for a real 127-qubit Eagle device.

Why this measurement has teeth
------------------------------
The strongest-looking quantum number in the oral track is in
``reports_e3_hybrid_fusion.json``: the fixed 8-qubit, zero-trainable-gate visual readout
reaches validation PR-AUC **0.868335** against **0.838846** for its matched PCA-8
classical control on identical rows through an identical logistic head -- a margin of
**+0.029489** in favour of the quantum representation. (It still fails the pre-registered
C7 gating null: increment +0.017116, ``survives_gating_null: false``.)

That margin was measured with no shot noise and no device error. Either it survives
realistic conditions or it does not, and the project has never asked. E8 asks.

Pre-registration
----------------
**Frozen before the first noisy circuit was executed. See PREREGISTRATION below.** The
rung set, the seed set, the shot count, the primary endpoint, the decision rule and the
integrity gates are module constants, not run-time arguments, so none of them can be
selected after seeing a result. The shot count is taken from
``Settings.QUANTUM_SHOTS`` -- it is the value the application was already configured
with, not a value chosen here.

**No branch of the decision rule licenses a quantum-advantage claim.** That is stated in
advance precisely so this phase cannot be read as a search for a positive. The best
available outcome is a *bounded* negative: a margin that survives noise but still fails
the gating null.

The three rungs
---------------
The 16 pre-registered observables are all Pauli-Z strings, hence diagonal in the
computational basis, so ``<Z_S> = sum_x p(x) (-1)^popcount(x & S)`` for *any* source of
the basis distribution ``p``. The three rungs differ **only** in where ``p`` comes from,
and share the identical contraction (``quantum_ml.readout.observable_expectations``) and
the identical downstream scorer:

* ``R1 ideal_exact``   -- ``p = |psi|**2`` from exact Aer statevectors. This is what every
  reported E2/E3 number used. Anchored: must reproduce 0.868335 to 1e-6 or the run halts.
* ``R2 ideal_shots``   -- ``p`` from shot frequencies, **noiseless** Aer, 1024 shots.
  Isolates finite-sampling error with device physics held out.
* ``R3 noisy_shots``   -- ``p`` from shot frequencies through the ``fake_sherbrooke``
  noise model, 1024 shots. Adds device error on top of R2.

Without R2 a degradation at R3 could not be attributed: shot noise and device noise would
be confounded. The ladder is what makes the attribution possible.

One subtlety, disclosed rather than buried: R2 executes the **logical** circuit (16 ``ry``,
16 ``cz``, depth 19 with measurement), while R3 executes the circuit **transpiled to the
device basis** (measured: 44 ``rz``, 35 ``sx``, 16 ``ecr``, 7 ``x``, depth 56). So
``R3 - R2`` is the cost of *running on this device*, which bundles calibrated gate/readout
error with the basis translation that running there requires. Those two are not separable
in any execution that actually happens on the hardware, so they are reported together
under that name rather than split by a model. Both ``transpile`` blocks are in the payload
so a reader can see exactly which circuit each rung ran.

**Basis, not layout.** R3 transpiles against ``AerSimulator(noise_model=...)``, which
inherits the noise model's ``basis_gates`` but is all-to-all: no coupling map is imposed,
so ``swaps_inserted`` is 0 because nothing had to route, *not* because the 8-qubit CZ ring
embeds in heavy-hex for free. It does not. Two measured consequences, both disclosed
because they bias in **opposite** directions and neither was pre-registered:

* *Optimistic.* Under the trivial layout the ring-closing pair ``(7, 0)`` is not physically
  connected on Sherbrooke, and ``NoiseModel.from_backend`` therefore carries no ``ecr``
  error for it. One of R3's 16 two-qubit gates ran **noise-free**, and the ring closed
  without paying routing. A layout-faithful transpile costs 8 SWAPs, 40 ``ecr`` and depth
  124 -- 2.2x R3's depth and 2.5x its two-qubit count.
* *Pessimistic.* The trivial layout puts logical qubit 6 on physical qubit 6, whose
  calibrated readout error is **0.257** against 0.011--0.061 for the other seven. The
  device cost is consequently dominated by one bad qubit: the three observables touching
  it deviate 0.191--0.221 while the other thirteen sit at 0.019--0.065. A layout-aware
  placement would avoid that qubit.

``layout_sensitivity`` answers this directly by rerunning R3 with the coupling map imposed.
It is **post-hoc** -- conceived after reading the frozen result -- and is reported
alongside the pre-registered rungs, never in place of them. ``DECISION_RULE`` is applied
to the pre-registered rungs only.

What this phase cannot claim
----------------------------
* **Not hardware.** A calibration-seeded noise model is a model. It omits drift,
  crosstalk beyond the calibrated pairs, and the queue-time recalibration a real job sees.
* ``device_calibrated_noise`` is reported from the backend resolution, not assumed. If
  ``qiskit-ibm-runtime`` is absent the module falls back to a synthetic depolarising model
  (``quantum_ml.backends.build_synthetic_noise_model``) and this phase **withholds the
  device claim** and labels every number synthetic.
* The test partition is never read. E8 reads TRAIN and VALIDATION only, through the same
  ``build_inputs`` path as E1/E2/E3.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings
from backend.evaluation.e0_controls import N_FOLDS, grouped_folds
from backend.evaluation.e1_failure_map import (
    BOOTSTRAP_DRAWS,
    PRIMARY,
    SEED,
    E1Inputs,
    _logistic_arm,
    _metrics,
    _standardise,
    bootstrap_pr_auc,
    build_inputs,
    candidate_matrices,
)
from backend.evaluation.e3_hybrid_fusion import (
    C5_DIM,
    C7_ANCHOR_TOLERANCE,
    C7_DIM,
    C7_REFERENCE_PR_AUC,
    MOBILENET_DIM,
    QUANTUM_VALIDATION_SAMPLE,
    Factory,
    Transform,
    _arm_summary,
    _make_pca_factory,
    _make_rff_factory,
    _score_arm,
)
from backend.ml.quantum_visual import QuantumVisualPreprocessor
from quantum_ml.backends import device_noise_model
from quantum_ml.readout import observable_expectations
from quantum_ml.visual_circuit import (
    E2_N_FEATURES,
    QuantumVisualFeatureMap,
    nn_ring_pairs,
    validate_against_aer,
)

logger = logging.getLogger(__name__)

#: Bump when the rung set, the endpoint or the decision rule changes. Persisted in the
#: payload so a stored result can never be reinterpreted under a different protocol.
E8_VERSION = "v1-e8-noisy-simulation-1"


# =============================================================================
# PREREGISTRATION -- frozen before the first noisy circuit was executed.
# =============================================================================

#: The three rungs, in order. Frozen: no rung may be added after seeing a result, and
#: none may be dropped. R2 exists solely to make a change at R3 attributable.
RUNGS: Tuple[str, ...] = ("ideal_exact", "ideal_shots", "noisy_shots")

#: Shot seeds for the two stochastic rungs. Three, frozen. Every shot-based headline is
#: reported as mean / SD / min / max across exactly these seeds -- never as a single draw,
#: and never as the best of them.
SHOT_SEEDS: Tuple[int, ...] = (42, 43, 44)

#: Secondary, attribution-only shot ladder for the feature-fidelity study. Declared here
#: so it is on the record that 1024 (the configured value) is the headline and these are
#: diagnostics. A headline may never be read off this ladder.
FIDELITY_SHOT_LADDER: Tuple[int, ...] = (256, 1024, 4096)

#: PRIMARY ENDPOINT. The margin of the quantum readout over its matched classical
#: control on identical rows through an identical head:
#:     margin(rung) = PR-AUC[quantum @ rung] - PR-AUC[pca]
#: measured at R1 and at R3. ``pca`` is classical and therefore identical at every rung,
#: so it is the fixed reference the margin is taken against.
PRIMARY_ENDPOINT = "margin_vs_pca8"

#: DECISION RULE, frozen. Evaluated by :func:`_verdict`, which is pure and takes only
#: measured numbers -- there is no branch a human selects after the fact.
DECISION_RULE: Tuple[str, ...] = (
    "If margin(R2) <= 0 while margin(R1) > 0: the R1 margin is destroyed by FINITE "
    "SAMPLING alone. Attribute it to the estimator, not to device physics. The noiseless "
    "margin is an artifact of exact expectation values that no real measurement can obtain.",
    "Else if margin(R3) <= 0: the R1 margin does NOT survive device noise. Report that the "
    "strongest-looking quantum number in the oral track is noiseless-only.",
    "Else if margin(R3) >= margin(R1) - sd(margin(R3)): the margin is NOISE-ROBUST. This is "
    "a bounded negative, not an advantage: the arm still fails the pre-registered C7 "
    "gating null (increment +0.017116, survives_gating_null=false, reports_e3_hybrid_fusion).",
    "Else: the margin is DEGRADED BUT POSITIVE. Report the measured shrinkage and the "
    "attribution split between R2 (sampling) and R3-minus-R2 (device error).",
)

#: Stated in advance. No measured outcome of this phase authorises an advantage claim.
NO_ADVANTAGE_CLAUSE = (
    "No branch of DECISION_RULE licenses a quantum-advantage claim. E8 can bound the "
    "noiseless margin; it cannot create a gating-null pass, it cannot substitute for "
    "hardware, and a noise-robust margin that fails the gating null is still a null."
)

#: Integrity anchors. Every one must reproduce or :class:`NoisySimulationError` is raised
#: and nothing is written -- the E6 pattern: halt rather than publish. Sources:
#: ``reports_e3_hybrid_fusion.json :: arms`` and ``:: c7_anchor``.
IDEAL_ANCHORS: Dict[str, float] = {
    "quantum": 0.868335,
    "pca": 0.838846,
    "rff": 0.795360,
}
ANCHOR_TOLERANCE = 1e-6

#: Recorded ideal entanglement witness, ``reports_e2_angle_experiment.json ::
#: selected.mean_abs_connected_correlation``. Reported against the noisy rungs: if device
#: noise drives the witness toward 0 then E2's "the CZ ring produced correlations a
#: separable encoding could not" is a noiseless-only statement.
IDEAL_WITNESS = 0.18379205

#: Convergence gate on the shot path. At the top of FIDELITY_SHOT_LADDER the shot-based
#: features must approach the exact ones at the 1/sqrt(S) rate this threshold encodes.
#: This is what *proves* the counts-to-basis-index convention is correct rather than
#: assuming it: a transposed bit order would not converge at all.
CONVERGENCE_SHOTS = 4096
CONVERGENCE_MAX_DEVIATION = 0.12  # ~5 sigma at S=4096 for a +-1-bounded observable


class NoisySimulationError(RuntimeError):
    """Raised when an E8 number cannot be produced on verified, patient-safe rows."""


# ---------------------------------------------------------------- shot-based readout
def _counts_to_probabilities(counts: Dict[str, int], n_qubits: int) -> np.ndarray:
    """Shot histogram -> length-``2**n_qubits`` probability vector.

    Qiskit renders bitstrings most-significant-qubit first and separates registers with
    spaces, so ``int(key, 2)`` is exactly the little-endian basis index that
    :func:`quantum_ml.readout.z_string_diagonal` indexes with (qubit 0 == bit 0). The
    convention is not assumed: :data:`CONVERGENCE_MAX_DEVIATION` gates it empirically,
    because a transposed bit order would not converge to the exact features at any S.
    """
    dim = 1 << n_qubits
    probabilities = np.zeros(dim, dtype=np.float64)
    total = 0
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        if not cleaned:
            continue
        probabilities[int(cleaned, 2)] += float(count)
        total += int(count)
    if total == 0:
        raise NoisySimulationError("Empty shot histogram; refusing to report a feature row.")
    return probabilities / float(total)


class ShotReadoutFeatureMap:
    """The E2 circuit read out from **shots** instead of an exact statevector.

    Deliberately exposes the same ``transform(angles) -> (n, 16)`` surface as
    :class:`~quantum_ml.visual_circuit.QuantumVisualFeatureMap`, so it drops into the
    unmodified ``e3_hybrid_fusion._make_quantum_factory`` /``_score_arm`` path. The
    circuit, the observable set and the contraction are the ideal map's own -- only the
    source of ``p`` changes. That is the whole point: a difference between rungs cannot
    be a difference of implementation.
    """

    def __init__(
        self,
        *,
        noise_model: Any,
        shots: int,
        seed: int,
        base: Optional[QuantumVisualFeatureMap] = None,
        optimization_level: int = 1,
        transpile_backend: Optional[Any] = None,
    ) -> None:
        from qiskit_aer import AerSimulator

        self._base = base or QuantumVisualFeatureMap()
        self.shots = int(shots)
        self.seed = int(seed)
        self.optimization_level = int(optimization_level)
        self._simulator = AerSimulator(noise_model=noise_model, seed_simulator=self.seed)
        # Transpiling against the device *backend* rather than the simulator is what
        # imposes layout and routing: AerSimulator is all-to-all even when it carries a
        # device noise model. Execution still happens on the noisy simulator, which applies
        # the calibrated errors by gate name and physical qubit index -- the standard
        # transpile-for-backend / run-on-noise-model workflow. Passing the coupling map to
        # AerSimulator directly is not equivalent and raises inside VF2Layout, because the
        # simulator's synthesised target has no per-qubit error map to average.
        self._transpile_target = transpile_backend if transpile_backend is not None else self._simulator
        self.routed = transpile_backend is not None
        self._transpile_stats: Dict[str, Any] = {}
        self.n_circuits_executed = 0

    @property
    def n_qubits(self) -> int:
        return self._base.n_qubits

    @property
    def observable_set(self) -> Any:
        return self._base.observable_set

    def transform(self, angles: np.ndarray) -> np.ndarray:
        """``(n, 16)`` shot-estimated observable expectations for a batch of angle rows."""
        from qiskit import transpile

        arr = self._base._prepare_angles(angles)
        if arr.shape[0] == 0:
            return np.empty((0, len(self._base.observable_set)), dtype=np.float64)

        circuits = []
        for row in arr:
            bound = self._base._bound(row)
            bound.measure_all()
            circuits.append(bound)
        transpiled = transpile(
            circuits,
            self._transpile_target,
            seed_transpiler=self.seed,
            optimization_level=self.optimization_level,
        )
        if not self._transpile_stats:
            self._transpile_stats = _transpile_summary(
                self._base.circuit, transpiled[0], self.optimization_level, self.routed
            )
        result = self._simulator.run(transpiled, shots=self.shots).result()
        self.n_circuits_executed += len(circuits)

        probabilities = np.empty((arr.shape[0], 1 << self.n_qubits), dtype=np.float64)
        for k in range(arr.shape[0]):
            probabilities[k] = _counts_to_probabilities(result.get_counts(k), self.n_qubits)
        return observable_expectations(probabilities, self._base.diagonals)

    def connected_correlations(self, features: np.ndarray) -> np.ndarray:
        return self._base.connected_correlations(features)

    def transpile_stats(self) -> Dict[str, Any]:
        return dict(self._transpile_stats)


def _transpile_summary(
    logical: Any, transpiled: Any, optimization_level: int, routed: bool
) -> Dict[str, Any]:
    """Measured cost of the circuit *as it would actually run* on the device basis.

    Reported rather than asserted. ``two_qubit_overhead`` is the routing cost: entangling
    gates beyond the 16 the logical ring needs. A nonzero value means the 8-qubit CZ ring
    does not embed in the heavy-hex coupling map and routing paid for it, which is a
    hardware-cost finding independent of any accuracy result.

    It is **not** measured by counting ``swap`` instructions. Routing inserts SWAPs, but
    the basis translator then decomposes each into three ``ecr`` gates, so ``count_ops``
    carries no ``swap`` key by the time the circuit is final -- a ``swaps_inserted`` field
    read that way would report 0 even for a fully routed circuit. The overhead is therefore
    derived from the entangling-gate count, which survives translation.
    """
    logical_ops = {str(k): int(v) for k, v in logical.count_ops().items()}
    device_ops = {str(k): int(v) for k, v in transpiled.count_ops().items()}
    two_qubit_names = {"ecr", "cx", "cz", "swap"}
    two_qubit = sum(count for gate, count in device_ops.items() if gate in two_qubit_names)
    logical_two_qubit = sum(
        count for gate, count in logical_ops.items() if gate in two_qubit_names
    )
    return {
        "optimization_level": int(optimization_level),
        "routing_applied": bool(routed),
        "coupling_map_imposed": bool(routed),
        "logical_depth": int(logical.depth()),
        "logical_ops": logical_ops,
        "logical_two_qubit_gates": int(logical_two_qubit),
        "device_depth": int(transpiled.depth()),
        "device_ops": device_ops,
        "device_two_qubit_gates": int(two_qubit),
        "two_qubit_overhead": int(two_qubit - logical_two_qubit),
        "depth_inflation": (
            round(transpiled.depth() / logical.depth(), 4) if logical.depth() else None
        ),
        "note": (
            "Depth and gate counts after transpilation to the noise model's basis "
            "(rz, sx, x, ecr)"
            + (
                " with the device coupling map imposed, so routing cost is included."
                if routed
                else ", with NO coupling map: AerSimulator(noise_model=...) is all-to-all, "
                "so no routing was required and two_qubit_overhead is 0 by construction, "
                "not because the ring embeds in heavy-hex. See layout_sensitivity."
            )
            + " Measured, not asserted."
        ),
    }


# ------------------------------------------------------------------------ statistics
def _summarise(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """mean / SD / min / max over the seed draws, with the draw list kept verbatim."""
    usable = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not usable:
        return {"measured": False, "n_seeds": 0, "draws": list(values)}
    arr = np.asarray(usable, dtype=np.float64)
    return {
        "measured": True,
        "n_seeds": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "sd": round(float(arr.std(ddof=1)), 6) if arr.size > 1 else 0.0,
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "draws": [round(float(v), 6) for v in usable],
    }


def _verdict(
    margin_r1: Optional[float],
    margin_r2: Dict[str, Any],
    margin_r3: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply :data:`DECISION_RULE` mechanically. Pure function of measured numbers."""
    if margin_r1 is None or not margin_r2.get("measured") or not margin_r3.get("measured"):
        return {
            "resolved": False,
            "verdict": "NOT MEASURED",
            "reason": "A rung failed to produce a margin; no branch may be selected.",
        }

    m1 = float(margin_r1)
    m2 = float(margin_r2["mean"])
    m3 = float(margin_r3["mean"])
    sd3 = float(margin_r3.get("sd") or 0.0)

    if m2 <= 0.0 < m1:
        label, rule = "SAMPLING-DESTROYED", 0
    elif m3 <= 0.0:
        label, rule = "NOISELESS-ONLY", 1
    elif m3 >= m1 - sd3:
        label, rule = "NOISE-ROBUST", 2
    else:
        label, rule = "DEGRADED-BUT-POSITIVE", 3

    return {
        "resolved": True,
        "verdict": label,
        "rule_index": rule,
        "rule_text": DECISION_RULE[rule],
        "margin_ideal_exact": round(m1, 6),
        "margin_ideal_shots_mean": round(m2, 6),
        "margin_noisy_shots_mean": round(m3, 6),
        "margin_noisy_shots_sd": round(sd3, 6),
        "sampling_cost": round(m2 - m1, 6),
        "device_cost": round(m3 - m2, 6),
        "total_cost": round(m3 - m1, 6),
        "cost_note": (
            "sampling_cost is R2-R1 (finite shots alone). device_cost is R3-R2 (calibrated "
            "device error on top). They sum to total_cost."
        ),
        "no_advantage_clause": NO_ADVANTAGE_CLAUSE,
    }


# ------------------------------------------------------------------- fidelity study
def _feature_fidelity(
    base: QuantumVisualFeatureMap,
    angles: np.ndarray,
    *,
    noise_model: Any,
    seed: int,
) -> Dict[str, Any]:
    """Per-observable deviation from exact, and the entanglement witness, per rung.

    Secondary and attribution-only. Runs the shot ladder on a *fixed* angle batch so the
    deviation is a property of the estimator and the device model, not of a resplit.
    """
    exact = base.transform(angles)
    exact_witness = float(np.mean(np.abs(base.connected_correlations(exact))))
    labels = list(base.observable_set.labels)

    out: Dict[str, Any] = {
        "n_rows": int(np.atleast_2d(angles).shape[0]),
        "shot_ladder": list(FIDELITY_SHOT_LADDER),
        "observable_labels": labels,
        "ideal_exact": {
            "mean_abs_connected_correlation": round(exact_witness, 8),
            "recorded_e2_witness": IDEAL_WITNESS,
            "witness_note": (
                "Recomputed here on the E8 angle batch; the recorded E2 value was selected "
                "under nested CV on TRAIN, so the two are close but not required equal."
            ),
        },
        "rungs": {},
    }

    for rung, model in (("ideal_shots", None), ("noisy_shots", noise_model)):
        per_shots: Dict[str, Any] = {}
        for shots in FIDELITY_SHOT_LADDER:
            fm = ShotReadoutFeatureMap(
                noise_model=model, shots=shots, seed=seed, base=base
            )
            features = fm.transform(angles)
            deviation = np.abs(features - exact)
            witness = float(np.mean(np.abs(fm.connected_correlations(features))))
            per_shots[str(shots)] = {
                "mean_abs_deviation": round(float(deviation.mean()), 6),
                "max_abs_deviation": round(float(deviation.max()), 6),
                "per_observable_mean_abs_deviation": {
                    labels[j]: round(float(deviation[:, j].mean()), 6)
                    for j in range(deviation.shape[1])
                },
                "mean_abs_connected_correlation": round(witness, 8),
                "witness_retained_fraction": (
                    round(witness / exact_witness, 6) if exact_witness else None
                ),
                "n_circuits_executed": int(fm.n_circuits_executed),
            }
        out["rungs"][rung] = per_shots

    return out


# ------------------------------------------------------------------ post-hoc sensitivity
def _device_backend(device: str) -> Tuple[Optional[Any], Optional[str]]:
    """The named fake backend itself, or ``(None, reason)``.

    Resolved by the same name rule ``quantum_ml.backends.device_noise_model`` uses, so a
    device that yields a calibrated noise model yields its backend too. The backend is what
    the post-hoc arm transpiles against, which is how layout and routing get imposed. Every
    failure degrades to a reason string: the arm is skipped, never faked.
    """
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError:
        return None, "qiskit-ibm-runtime not installed"
    class_name = "".join(part.capitalize() for part in device.split("_"))
    backend_class = getattr(fake_provider, class_name, None)
    if backend_class is None:
        return None, f"unknown device {device!r}"
    try:
        return backend_class(), None
    except Exception as exc:  # noqa: BLE001 - skip the arm, do not fail the phase
        return None, f"could not construct {device}: {exc}"


def _ring_pair_calibration(device: str, n_qubits: int) -> Dict[str, Any]:
    """Which E2 ring pairs the trivial layout puts on *uncalibrated* couplings.

    This is the measurement behind the "optimistic" half of the disclosure: a ring pair
    that is not physically coupled carries no ``ecr`` error in the noise model, so that
    entangling gate runs noise-free under the pre-registered R3.
    """
    out: Dict[str, Any] = {"measured": False}
    try:
        from qiskit_ibm_runtime import fake_provider
        from qiskit_aer.noise import NoiseModel as _NoiseModel
    except ImportError:
        out["reason"] = "qiskit-ibm-runtime not installed"
        return out
    class_name = "".join(part.capitalize() for part in device.split("_"))
    backend_class = getattr(fake_provider, class_name, None)
    if backend_class is None:
        out["reason"] = f"unknown device {device!r}"
        return out
    try:
        backend = backend_class()
        coupled = set()
        for a, b in backend.coupling_map:
            coupled.add((int(a), int(b)))
        model = _NoiseModel.from_backend(backend)
        ecr_errors = getattr(model, "_local_quantum_errors", {}).get("ecr", {})
        properties = backend.properties()
        pairs = []
        for a, b in nn_ring_pairs(n_qubits):
            connected = (a, b) in coupled or (b, a) in coupled
            has_error = (a, b) in ecr_errors or (b, a) in ecr_errors
            pairs.append(
                {
                    "pair": [int(a), int(b)],
                    "physically_coupled": bool(connected),
                    "ecr_error_in_noise_model": bool(has_error),
                }
            )
        readout = {
            f"q{q}": round(float(properties.readout_error(q)), 6) for q in range(n_qubits)
        }
        worst = max(readout, key=lambda k: readout[k])
        out.update(
            {
                "measured": True,
                "ring_pairs": pairs,
                "n_pairs_without_calibrated_error": sum(
                    1 for p in pairs if not p["ecr_error_in_noise_model"]
                ),
                "readout_error_by_physical_qubit": readout,
                "worst_readout_qubit": worst,
                "worst_readout_error": readout[worst],
                "note": (
                    "Trivial layout: logical qubit i -> physical qubit i. A ring pair with "
                    "ecr_error_in_noise_model=false ran noise-free in the pre-registered "
                    "R3 (optimistic). worst_readout_qubit carries the degradation seen in "
                    "feature_fidelity's per-observable deviations (pessimistic)."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report the reason, do not fail the phase
        out["reason"] = f"calibration inspection failed: {exc}"
    return out


def _layout_sensitivity(
    *,
    configured_device: str,
    device_calibrated: bool,
    noise_model: Any,
    base_map: QuantumVisualFeatureMap,
    mobilenet_train: np.ndarray,
    mobilenet_validation: np.ndarray,
    inputs: Any,
    folds: Any,
    shots: int,
    shot_seeds: Sequence[int],
    seed: int,
    bootstrap_draws: int,
    pca_reference: Optional[float],
    noisy_margin: Dict[str, Any],
    quantum_factory_for: Callable[[Any], Factory],
) -> Dict[str, Any]:
    """R3 rerun with the device coupling map imposed. **Post-hoc, not pre-registered.**

    The pre-registered R3 transpiles to the noise model's basis but not its connectivity,
    because ``AerSimulator(noise_model=...)`` is all-to-all. This arm imposes the coupling
    map so routing is paid for, and reports the result whichever way it comes out.

    It cannot change ``verdict``, which was computed before this ran. Its purpose is to
    bound how much of the pre-registered margin depends on the missing routing cost.
    """
    out: Dict[str, Any] = {
        "preregistered": False,
        "post_hoc": True,
        "post_hoc_reason": (
            "Conceived after reading the frozen result, when R3's two_qubit_overhead of 0 "
            "proved to mean 'no coupling map imposed' rather than 'the ring embeds in "
            "heavy-hex for free'. Reported alongside the pre-registered rungs; excluded "
            "from DECISION_RULE, which had already been applied."
        ),
        "excluded_from_verdict": True,
        "calibration": _ring_pair_calibration(configured_device, base_map.n_qubits),
    }
    if not device_calibrated:
        out["skipped"] = "noise model is the synthetic fallback; it has no coupling map"
        return out
    backend, reason = _device_backend(configured_device)
    if backend is None:
        out["skipped"] = reason
        return out

    per_seed_pr_auc: List[Optional[float]] = []
    per_seed: Dict[str, Any] = {}
    transpile_stats: Dict[str, Any] = {}
    executed = 0
    for shot_seed in shot_seeds:
        fm = ShotReadoutFeatureMap(
            noise_model=noise_model,
            shots=shots,
            seed=int(shot_seed),
            base=base_map,
            transpile_backend=backend,
        )
        fitted = _score_arm(
            mobilenet_train,
            mobilenet_validation,
            inputs.y_train,
            folds,
            preprocess=quantum_factory_for(fm),
            seed=seed,
        )
        summary = _arm_summary(
            f"quantum_routed_s{shot_seed}",
            fitted,
            inputs,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
        )
        per_seed_pr_auc.append(summary.get("validation_pr_auc"))
        per_seed[str(shot_seed)] = {
            "validation_pr_auc": summary.get("validation_pr_auc"),
            "train_cv_pr_auc": fitted.get("train_cv_pr_auc"),
            "selected_c": fitted.get("selected_c"),
            "validation_bootstrap": summary.get("validation_bootstrap"),
            "n_circuits_executed": int(fm.n_circuits_executed),
        }
        executed += int(fm.n_circuits_executed)
        if not transpile_stats:
            transpile_stats = fm.transpile_stats()
        logger.info(
            "E8 routed_noisy_shots seed %d: validation PR-AUC %s",
            shot_seed,
            summary.get("validation_pr_auc"),
        )

    margins = [
        None if v is None or pca_reference is None else float(v) - float(pca_reference)
        for v in per_seed_pr_auc
    ]
    routed_margin = _summarise(margins)
    unrouted_mean = noisy_margin.get("mean")
    routed_mean = routed_margin.get("mean")
    out.update(
        {
            "shots": shots,
            "shot_seeds": list(int(s) for s in shot_seeds),
            "per_seed": per_seed,
            "validation_pr_auc": _summarise(per_seed_pr_auc),
            PRIMARY_ENDPOINT: routed_margin,
            "transpile": transpile_stats,
            "n_circuits_executed": executed,
            "routing_cost_on_margin": (
                None
                if unrouted_mean is None or routed_mean is None
                else round(float(routed_mean) - float(unrouted_mean), 6)
            ),
            "interpretation": (
                "routing_cost_on_margin is (routed R3 margin) - (pre-registered R3 "
                "margin). Negative means the missing routing cost was flattering the "
                "pre-registered result by that much. It adjusts no pre-registered number "
                "and licenses no advantage claim."
            ),
        }
    )
    return out


# -------------------------------------------------------------------------- the phase
def run(
    *,
    config: Optional[Settings] = None,
    seed: int = SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    shot_seeds: Sequence[int] = SHOT_SEEDS,
    fidelity_rows: int = 64,
) -> Dict[str, Any]:
    """Run E8 on the primary leak-free condition. Writes nothing; ``main`` persists it."""
    started = time.perf_counter()
    cfg = config or Settings()
    shots = int(cfg.QUANTUM_SHOTS)

    # --- The noise model, and an honest statement of which one we actually got.
    noise_model, noise_description, device_calibrated = device_noise_model(
        cfg.QUANTUM_NOISE_MODEL
    )
    noise_block = {
        "configured_device": cfg.QUANTUM_NOISE_MODEL,
        "resolved_description": noise_description,
        "device_calibrated_noise": bool(device_calibrated),
        "shots": shots,
        "shots_source": "Settings.QUANTUM_SHOTS (the configured value, not chosen by E8)",
        "basis_gates": sorted(str(g) for g in noise_model.basis_gates),
        "noise_instructions": sorted(str(i) for i in noise_model.noise_instructions),
        "n_noisy_qubits": len(noise_model.noise_qubits),
        "claim_withheld": None
        if device_calibrated
        else (
            "qiskit-ibm-runtime was unavailable, so this run used the synthetic "
            "depolarising fallback. Every number below is SYNTHETIC-NOISE. The "
            "device-calibrated claim is withheld."
        ),
        "not_hardware": (
            "A calibration-seeded Aer noise model is a model, not a device. It omits "
            "drift, uncalibrated crosstalk and queue-time recalibration. No circuit in "
            "this repository has ever run on quantum hardware."
        ),
    }
    logger.info(
        "E8 noise model: %s (device_calibrated=%s), %d shots, %d noisy qubits",
        noise_description,
        device_calibrated,
        shots,
        len(noise_model.noise_qubits),
    )

    # --- Rows. TRAIN and VALIDATION only, through the same path as E1/E2/E3.
    per_condition, provenance = build_inputs(config=cfg, conditions=(PRIMARY,))
    inputs: E1Inputs = per_condition[PRIMARY]
    if set(inputs.patients_train) & set(inputs.patients_validation):
        raise NoisySimulationError(
            "Patient overlap between train and validation; build_inputs should have "
            "refused. Refusing to measure."
        )

    folds, cv_note = grouped_folds(
        inputs.y_train, inputs.patients_train, seed=seed, n_splits=N_FOLDS
    )

    specs = candidate_matrices(inputs)
    c6_train, c6_validation = specs["C6"]["train"], specs["C6"]["validation"]
    c5_train, c5_validation = specs["C5"]["train"], specs["C5"]["validation"]
    if c6_train.shape[1] != MOBILENET_DIM or c5_train.shape[1] != C5_DIM:
        raise NoisySimulationError(
            f"Unexpected C6/C5 dimensions ({c6_train.shape[1]}/{c5_train.shape[1]}); "
            f"expected {MOBILENET_DIM}/{C5_DIM}. Cache is stale or misassembled."
        )
    c7_train = np.hstack([c6_train, c5_train]).astype(np.float64)
    c7_validation = np.hstack([c6_validation, c5_validation]).astype(np.float64)
    if c7_train.shape[1] != C7_DIM:
        raise NoisySimulationError(
            f"C7 has {c7_train.shape[1]} columns, expected {C7_DIM}."
        )

    mobilenet_train = np.asarray(inputs.e1_train["mobilenet"], dtype=np.float64)
    mobilenet_validation = np.asarray(inputs.e1_validation["mobilenet"], dtype=np.float64)

    rows_block = {
        "n_train": int(inputs.y_train.size),
        "n_validation": int(inputs.y_validation.size),
        "n_train_positive": int(inputs.y_train.sum()),
        "n_validation_positive": int(inputs.y_validation.sum()),
        "n_train_patients": len(set(inputs.patients_train)),
        "n_validation_patients": len(set(inputs.patients_validation)),
        "patient_overlap_train_validation": 0,
    }

    # --- ANCHOR 1: C7 through E1's own scoring path.
    c7_xt, c7_xv = _standardise(c7_train, c7_validation)
    c7_fit = _logistic_arm(c7_xt, inputs.y_train, c7_xv, folds, seed=seed)
    c7_metrics = _metrics(inputs.y_validation, c7_fit["validation_probability"])
    c7_pr_auc = None if not c7_metrics else c7_metrics.get("pr_auc")
    c7_ok = (
        c7_pr_auc is not None
        and abs(float(c7_pr_auc) - C7_REFERENCE_PR_AUC) <= C7_ANCHOR_TOLERANCE
    )

    # --- The exact feature map, validated against Aer before any quantum number is used.
    base_map = QuantumVisualFeatureMap()
    pre_all = QuantumVisualPreprocessor.fit(mobilenet_train, seed=seed)
    sample = min(int(QUANTUM_VALIDATION_SAMPLE), mobilenet_train.shape[0])
    aer_check = validate_against_aer(base_map, pre_all.to_angles(mobilenet_train[:sample]))

    def _quantum_factory_for(feature_map: Any) -> Factory:
        """``_make_quantum_factory`` with a swappable readout. Fold-honest: the
        preprocessor is refit on the fold-train source only; the map has 0 trainable
        parameters and is fold-independent."""

        def factory(fold_source: np.ndarray) -> Transform:
            pre = QuantumVisualPreprocessor.fit(fold_source, seed=seed)
            return lambda x: feature_map.transform(pre.to_angles(x))

        return factory

    # --- R1 and the two classical controls. All through the unmodified E3 scorer.
    arms: Dict[str, Any] = {}
    ideal_pr_auc: Dict[str, Optional[float]] = {}
    for name, factory in (
        ("quantum", _quantum_factory_for(base_map)),
        ("pca", _make_pca_factory(seed=seed)),
        ("rff", _make_rff_factory(seed=seed, n_features=E2_N_FEATURES)),
    ):
        fitted = _score_arm(
            mobilenet_train,
            mobilenet_validation,
            inputs.y_train,
            folds,
            preprocess=factory,
            seed=seed,
        )
        summary = _arm_summary(
            name, fitted, inputs, bootstrap_draws=bootstrap_draws, seed=seed
        )
        arms[name] = summary
        ideal_pr_auc[name] = summary.get("validation_pr_auc")
        logger.info("E8 R1 %s: validation PR-AUC %s", name, summary.get("validation_pr_auc"))

    # --- Integrity gate: halt rather than publish.
    anchor_report: Dict[str, Any] = {
        "c7": {
            "target": C7_REFERENCE_PR_AUC,
            "reproduced": None if c7_pr_auc is None else round(float(c7_pr_auc), 6),
            "tolerance": C7_ANCHOR_TOLERANCE,
            "ok": bool(c7_ok),
        }
    }
    failures: List[str] = [] if c7_ok else ["c7"]
    for name, target in IDEAL_ANCHORS.items():
        got = ideal_pr_auc.get(name)
        ok = got is not None and abs(float(got) - target) <= ANCHOR_TOLERANCE
        anchor_report[name] = {
            "target": target,
            "reproduced": None if got is None else round(float(got), 6),
            "tolerance": ANCHOR_TOLERANCE,
            "ok": bool(ok),
        }
        if not ok:
            failures.append(name)
    anchor_report["all_ok"] = not failures
    anchor_report["source"] = (
        "reports_e3_hybrid_fusion.json :: arms.{quantum,pca,rff}.validation_pr_auc and "
        ":: c7_anchor.target_pr_auc"
    )
    if failures:
        raise NoisySimulationError(
            "E8 refuses to report a noisy comparison whose ideal baseline does not "
            f"reproduce. Failed anchors: {', '.join(failures)}. "
            f"Measured: { {k: anchor_report[k].get('reproduced') for k in failures} }. "
            "A noisy number is only interpretable against a verified noiseless one."
        )

    # --- R2 and R3: the same arm, the same scorer, shots instead of amplitudes.
    shot_rungs: Dict[str, Any] = {}
    transpile_stats: Dict[str, Any] = {}
    circuits_executed = 0
    for rung, model in (("ideal_shots", None), ("noisy_shots", noise_model)):
        per_seed_pr_auc: List[Optional[float]] = []
        per_seed_train_cv: List[Optional[float]] = []
        per_seed: Dict[str, Any] = {}
        for shot_seed in shot_seeds:
            fm = ShotReadoutFeatureMap(
                noise_model=model, shots=shots, seed=int(shot_seed), base=base_map
            )
            fitted = _score_arm(
                mobilenet_train,
                mobilenet_validation,
                inputs.y_train,
                folds,
                preprocess=_quantum_factory_for(fm),
                seed=seed,
            )
            summary = _arm_summary(
                f"quantum_{rung}_s{shot_seed}",
                fitted,
                inputs,
                bootstrap_draws=bootstrap_draws,
                seed=seed,
            )
            per_seed_pr_auc.append(summary.get("validation_pr_auc"))
            per_seed_train_cv.append(fitted.get("train_cv_pr_auc"))
            per_seed[str(shot_seed)] = {
                "validation_pr_auc": summary.get("validation_pr_auc"),
                "validation_roc_auc": summary.get("validation_roc_auc"),
                "train_cv_pr_auc": fitted.get("train_cv_pr_auc"),
                "selected_c": fitted.get("selected_c"),
                "validation_bootstrap": summary.get("validation_bootstrap"),
                "n_circuits_executed": int(fm.n_circuits_executed),
            }
            circuits_executed += fm.n_circuits_executed
            if rung not in transpile_stats:
                transpile_stats[rung] = fm.transpile_stats()
            logger.info(
                "E8 %s seed %d: validation PR-AUC %s",
                rung,
                shot_seed,
                summary.get("validation_pr_auc"),
            )

        margins = [
            None if v is None else float(v) - float(ideal_pr_auc["pca"])
            for v in per_seed_pr_auc
        ]
        shot_rungs[rung] = {
            "shots": shots,
            "shot_seeds": list(int(s) for s in shot_seeds),
            "per_seed": per_seed,
            "validation_pr_auc": _summarise(per_seed_pr_auc),
            "train_cv_pr_auc": _summarise(per_seed_train_cv),
            PRIMARY_ENDPOINT: _summarise(margins),
            "transpile": transpile_stats.get(rung, {}),
        }

    margin_r1 = (
        None
        if ideal_pr_auc.get("quantum") is None or ideal_pr_auc.get("pca") is None
        else float(ideal_pr_auc["quantum"]) - float(ideal_pr_auc["pca"])
    )
    verdict = _verdict(
        margin_r1,
        shot_rungs["ideal_shots"][PRIMARY_ENDPOINT],
        shot_rungs["noisy_shots"][PRIMARY_ENDPOINT],
    )

    # --- POST-HOC: the same noisy rung with the device coupling map imposed.
    #
    # Conceived after reading the frozen result, when the pre-registered R3's
    # two_qubit_overhead of 0 turned out to mean "no coupling map" rather than "the ring
    # embeds for free". Reported alongside the pre-registered rungs and excluded from
    # DECISION_RULE, which was already applied above.
    layout_sensitivity = _layout_sensitivity(
        configured_device=cfg.QUANTUM_NOISE_MODEL,
        device_calibrated=device_calibrated,
        noise_model=noise_model,
        base_map=base_map,
        mobilenet_train=mobilenet_train,
        mobilenet_validation=mobilenet_validation,
        inputs=inputs,
        folds=folds,
        shots=shots,
        shot_seeds=shot_seeds,
        seed=seed,
        bootstrap_draws=bootstrap_draws,
        pca_reference=ideal_pr_auc.get("pca"),
        noisy_margin=shot_rungs["noisy_shots"][PRIMARY_ENDPOINT],
        quantum_factory_for=_quantum_factory_for,
    )
    circuits_executed += int(layout_sensitivity.get("n_circuits_executed", 0) or 0)

    # --- Secondary: feature fidelity, witness survival, shot ladder.
    n_fid = min(int(fidelity_rows), mobilenet_train.shape[0])
    fidelity = _feature_fidelity(
        base_map,
        pre_all.to_angles(mobilenet_train[:n_fid]),
        noise_model=noise_model,
        seed=int(shot_seeds[0]),
    )

    # --- Convergence gate on the counts->index convention.
    top = fidelity["rungs"]["ideal_shots"][str(CONVERGENCE_SHOTS)]
    converged = float(top["max_abs_deviation"]) <= CONVERGENCE_MAX_DEVIATION
    convergence = {
        "shots": CONVERGENCE_SHOTS,
        "max_abs_deviation": top["max_abs_deviation"],
        "threshold": CONVERGENCE_MAX_DEVIATION,
        "passed": bool(converged),
        "what_it_proves": (
            "The shot path converges to the exact path as S grows, which verifies the "
            "counts-to-basis-index convention empirically. A transposed bit order would "
            "not converge at any S."
        ),
    }
    if not converged:
        raise NoisySimulationError(
            f"Shot readout did not converge to the exact readout at {CONVERGENCE_SHOTS} "
            f"shots (max deviation {top['max_abs_deviation']} > "
            f"{CONVERGENCE_MAX_DEVIATION}). The readout path is wrong; refusing to report."
        )

    duration = round(time.perf_counter() - started, 1)
    return {
        "e8_version": E8_VERSION,
        "question": (
            "Does the E2/E3 quantum visual readout's +0.029489 PR-AUC margin over its "
            "matched PCA-8 classical control survive finite sampling and calibrated "
            "device noise, or is it an artifact of exact noiseless simulation?"
        ),
        "test_partition_used": False,
        "test_partition_note": (
            "Reads TRAIN and VALIDATION only via e1_failure_map.build_inputs, which "
            "iterates only ('train', 'validation'). No code path in this module selects "
            "the test partition."
        ),
        "selection_performed": False,
        "selection_note": (
            "No model, rung, seed or shot count was selected on any measured outcome. "
            "RUNGS, SHOT_SEEDS, FIDELITY_SHOT_LADDER, PRIMARY_ENDPOINT and DECISION_RULE "
            "are module constants frozen before the first noisy circuit executed; shots "
            "comes from Settings.QUANTUM_SHOTS."
        ),
        "primary_condition": PRIMARY,
        "cv_note": cv_note,
        "rows": rows_block,
        "noise": noise_block,
        "anchors": anchor_report,
        "quantum_feature_map_aer_check": aer_check,
        "convergence_gate": convergence,
        "preregistration": {
            "rungs": list(RUNGS),
            "shot_seeds": list(int(s) for s in SHOT_SEEDS),
            "fidelity_shot_ladder": list(FIDELITY_SHOT_LADDER),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "primary_endpoint_definition": (
                "margin(rung) = PR-AUC[quantum @ rung] - PR-AUC[pca]. pca is classical and "
                "identical at every rung, so it is the fixed reference."
            ),
            "decision_rule": list(DECISION_RULE),
            "no_advantage_clause": NO_ADVANTAGE_CLAUSE,
            "frozen_before_first_noisy_circuit": True,
        },
        "ideal_exact": {
            "arms": arms,
            PRIMARY_ENDPOINT: None if margin_r1 is None else round(margin_r1, 6),
            "note": (
                "R1 reproduces reports_e3_hybrid_fusion.json exactly; it is the baseline "
                "the noisy rungs are read against, not a new result."
            ),
        },
        "shot_rungs": shot_rungs,
        "verdict": verdict,
        "layout_sensitivity": layout_sensitivity,
        "feature_fidelity": fidelity,
        "cost": {
            "n_noisy_and_shot_circuits_executed": int(circuits_executed),
            "note": (
                "Circuits submitted to Aer across both shot rungs, all seeds and the "
                "post-hoc routed arm for the scored arm. The fidelity study's circuits are "
                "counted separately within feature_fidelity."
            ),
        },
        "provenance": provenance,
        "protocol": {
            "scorer": (
                "e3_hybrid_fusion._score_arm (unmodified) + _arm_summary; "
                "e1_failure_map._standardise / _cv_select / bootstrap_pr_auc"
            ),
            "folds": f"{N_FOLDS}-fold StratifiedGroupKFold on TRAIN grouped by patient",
            "what_differs_between_rungs": (
                "Only the source of the basis distribution p. Circuit, observable set, "
                "contraction, standardisation, head, folds and seeds are identical."
            ),
            "rung_circuit_disclosure": (
                "R2 runs the logical circuit (no basis translation); R3 runs it transpiled "
                "to the noise model's BASIS ONLY. No coupling map is imposed at either "
                "rung, because AerSimulator(noise_model=...) is all-to-all -- so R3's "
                "two_qubit_overhead of 0 means nothing had to route, NOT that the 8-qubit "
                "CZ ring embeds in heavy-hex for free. It does not. So device_cost = R3 - R2 "
                "is the cost of running on this device's basis with its calibrated "
                "gate/readout error, excluding routing. Those two are inseparable in any "
                "real execution, so they are reported together rather than split by a "
                "model. The routing cost is measured separately and post-hoc in "
                "layout_sensitivity. See shot_rungs.*.transpile."
            ),
            "spec": "report/PAPER_MASTER_EVIDENCE_SHEET.md (noisy-simulation deduction)",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "duration_seconds": duration,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E8 noisy-simulation replication.")
    parser.add_argument(
        "--out",
        default="backend/artifacts/reports/reports_e8_noisy_simulation.json",
        help="Where to write the payload.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--fidelity-rows", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Qiskit's pass manager logs one line per transpiler pass per circuit at INFO. This
    # phase transpiles ~10k circuits, which buried the phase's own 20 lines under 16 MB of
    # pass timings on the first run. Raise only that logger; nothing else is suppressed.
    logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)

    from pathlib import Path

    payload = run(
        bootstrap_draws=int(args.bootstrap_draws),
        fidelity_rows=int(args.fidelity_rows),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("E8 verdict: %s", payload["verdict"].get("verdict"))
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
