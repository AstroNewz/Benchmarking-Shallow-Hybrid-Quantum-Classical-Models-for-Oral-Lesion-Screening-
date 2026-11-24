"""E7 -- oral-track PCA component ladder (POST-HOC).

§8.4 of the evidence sheet records a gap: the oral quantum track reduces
MobileNet-576 to **8** PCA components, that 8 was *predetermined* by
``QUANTUM_QUBITS = 8`` and ``qubits_for_dimension``'s power-of-two requirement, and
**no artifact anywhere sweeps the number of components on the oral track**. The only
reduction ladder that exists is the PTB-XL ECG one, which is a different dataset and a
different modality. That absence is what forces the evidence sheet to say "16D is
theoretically optimal is unsupported" on the weak ground of *no evidence* rather than
the strong ground of *evidence against*.

This module measures the ladder. Two representations, each reduced to
k in (4, 8, 16, 32, 64, 128, 256) and also scored unreduced:

* ``mobilenet`` -- the frozen MobileNetV3-Small embedding (576-d). This is **exactly
  what E2/E3 reduce**, so its k = 8 rung is the oral quantum track's own operating
  point and its k = 16 rung is the dimension the brief asks about.
* ``c7`` -- the 1,195-d C7 fusion (embedding + CIELAB descriptor + multiscale), the
  project's best classical representation.

PCA comes from ``backend/ml/reduction.py`` -- the same ``DimensionalityReducer`` the
deployed pipeline uses, fitted on the **frozen TRAIN partition only** and applied to
validation, so the ladder measures the pipeline's own reduction rather than a
reimplementation of it. Every rung is scored through E1's ``run_candidate`` on E1's
patient-grouped folds with the E5 uniform arm grid, so rungs are comparable to each
other and to the E5 lattice.

POST-HOC. Not pre-registered: ``preregistered: false``. No p-value is reported. The
ladder's purpose is to answer one question with a measurement -- *is 16 special?* --
and the answer it returns is whatever it returns.

Run::

    python -m backend.evaluation.e7_dimension_ladder
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.e0_controls import grouped_folds
from backend.evaluation.e1_failure_map import (
    BOOTSTRAP_DRAWS,
    PRIMARY,
    SEED,
    E1Inputs,
    bootstrap_pr_auc,
    build_inputs,
    run_candidate,
)
from backend.evaluation.e5_posthoc_ablation import UNIFORM_ARMS, _registered_arms
from backend.ml.artifacts import ArtifactStore
from backend.ml.reduction import DimensionalityReducer, qubits_for_dimension

logger = logging.getLogger(__name__)

LADDER_VERSION = "v1-e7-dimension-ladder-1"
COMPONENTS: Tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
REPRESENTATIONS: Tuple[Tuple[str, str], ...] = (
    (
        "mobilenet",
        "frozen MobileNetV3-Small embedding (576) -- the matrix E2/E3 reduce to PCA-8",
    ),
    (
        "c7",
        "C7 fusion: embedding (576) + CIELAB descriptor (163) + multiscale (456) = 1195",
    ),
)


class LadderError(RuntimeError):
    """Raised when a ladder rung cannot be built as specified."""


def _representation(inputs: E1Inputs, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(train, validation)`` for one named representation, from E1's own inputs."""
    if name == "mobilenet":
        return inputs.e1_train["mobilenet"], inputs.e1_validation["mobilenet"]
    if name == "c7":
        return (
            np.hstack(
                [
                    inputs.e1_train["mobilenet"],
                    inputs.descriptor_train,
                    inputs.e1_train["multiscale"],
                ]
            ),
            np.hstack(
                [
                    inputs.e1_validation["mobilenet"],
                    inputs.descriptor_validation,
                    inputs.e1_validation["multiscale"],
                ]
            ),
        )
    raise LadderError(f"Unknown representation {name!r}.")


def _reduce(
    train: np.ndarray, validation: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """PCA-k via the deployed ``DimensionalityReducer``, fitted on train only."""
    reducer = DimensionalityReducer(target_dimension=int(k))
    reducer.fit(train)
    reduced_train = reducer.transform(train)
    reduced_validation = reducer.transform(validation)
    summary = reducer.summary()
    ratio = reducer.explained_variance_ratio
    return (
        reduced_train,
        reduced_validation,
        {
            "method": summary.get("method"),
            "n_components": int(reducer.n_components),
            "n_features_in": int(reducer.n_features_in),
            "n_samples_seen": int(reducer.n_samples_seen),
            "fitted_on": "train partition only",
            "explained_variance_ratio_sum": (
                round(float(np.sum(ratio)), 6) if ratio is not None else None
            ),
            "whiten": bool(reducer.whiten),
            "reduction_version": summary.get("version"),
        },
    )


def run_ladder(
    *,
    config: Optional[Settings] = None,
    representations: Sequence[str] = tuple(name for name, _ in REPRESENTATIONS),
    components: Sequence[int] = COMPONENTS,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Score every (representation, k) rung on the frozen primary condition."""
    cfg = config or default_settings
    per_condition, provenance = build_inputs(config=cfg, conditions=(PRIMARY,))
    inputs = per_condition[PRIMARY]
    folds, cv_note = grouped_folds(inputs.y_train, inputs.patients_train, seed=seed)
    labels = {name: label for name, label in REPRESENTATIONS}

    out: Dict[str, Any] = {}
    for rep in representations:
        train, validation = _representation(inputs, rep)
        width = int(train.shape[1])
        rungs: List[Dict[str, Any]] = []
        # k values above the matrix width, or above the number of training rows, are not
        # a reduction -- they are skipped and recorded as skipped rather than silently
        # collapsing onto the unreduced rung.
        ladder: List[Optional[int]] = [
            k for k in components if int(k) < width and int(k) <= int(train.shape[0])
        ] + [None]
        for k in ladder:
            if k is None:
                x_train, x_validation = train, validation
                reduction = {
                    "method": "identity",
                    "n_components": width,
                    "n_features_in": width,
                    "fitted_on": "n/a (unreduced)",
                    "explained_variance_ratio_sum": 1.0,
                }
                arm_name = f"L7_{rep}_full"
            else:
                x_train, x_validation, reduction = _reduce(train, validation, int(k))
                arm_name = f"L7_{rep}_k{int(k)}"

            spec = {
                "train": x_train,
                "validation": x_validation,
                "label": (
                    f"{labels[rep]} -> "
                    + ("unreduced" if k is None else f"PCA-{int(k)}")
                ),
                "gap_tested": None,
                "arms": list(UNIFORM_ARMS),
            }
            started = time.perf_counter()
            with _registered_arms([arm_name], UNIFORM_ARMS):
                result = run_candidate(arm_name, spec, inputs, folds, seed=seed)
            internal = result.pop("_arms_internal")
            best_arm = result.get("best_arm")
            if best_arm is not None:
                scores = internal[best_arm]["_validation_probability"]
                if scores is None:
                    scores = internal[best_arm]["_validation_score"]
                interval = bootstrap_pr_auc(
                    inputs.y_validation,
                    np.asarray(scores, dtype=np.float64),
                    inputs.patients_validation,
                    draws=draws,
                    seed=seed,
                )
            else:
                interval = {
                    "measured": False,
                    "reason": "no arm of this rung produced a validation PR-AUC",
                }
            arms_block = result.get("arms") or {}
            rungs.append(
                {
                    "k": None if k is None else int(k),
                    "dimension": int(x_train.shape[1]),
                    "qubits_if_amplitude_encoded": (
                        qubits_for_dimension(int(x_train.shape[1]))
                        if int(x_train.shape[1]) in {2**b for b in range(1, 20)}
                        else None
                    ),
                    "reduction": reduction,
                    "validation_pr_auc": result.get("best_validation_pr_auc"),
                    "best_arm": best_arm,
                    "best_linear_validation_pr_auc": result.get(
                        "best_linear_validation_pr_auc"
                    ),
                    "best_nonlinear_validation_pr_auc": result.get(
                        "best_nonlinear_validation_pr_auc"
                    ),
                    "per_arm_validation_pr_auc": {
                        arm: payload.get("validation_pr_auc")
                        for arm, payload in arms_block.items()
                    },
                    "per_arm_train_cv_pr_auc": {
                        arm: payload.get("train_cv_pr_auc")
                        for arm, payload in arms_block.items()
                    },
                    "arm_failures": result.get("arm_failures") or {},
                    "bootstrap": interval,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
            logger.info(
                "  %s k=%s dim=%d -> validation PR-AUC %s (%s)",
                rep,
                k,
                int(x_train.shape[1]),
                result.get("best_validation_pr_auc"),
                best_arm,
            )

        out[rep] = {
            "representation": rep,
            "label": labels[rep],
            "native_dimension": width,
            "skipped_k": [
                int(k)
                for k in components
                if not (int(k) < width and int(k) <= int(train.shape[0]))
            ],
            "rungs": rungs,
            "verdict": _ladder_verdict(rungs),
        }
    return out, {"cv_note": cv_note, "provenance": provenance}


def _ladder_verdict(rungs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the ladder says about k = 8 and k = 16, computed rather than asserted."""
    scored = [
        r for r in rungs if isinstance(r.get("validation_pr_auc"), (int, float))
    ]
    if not scored:
        return {"measured": False, "reason": "no rung produced a validation PR-AUC"}
    by_k = {r["k"]: float(r["validation_pr_auc"]) for r in scored}
    best = max(scored, key=lambda r: float(r["validation_pr_auc"]))
    values = [float(r["validation_pr_auc"]) for r in scored]
    reduced = [r for r in scored if r["k"] is not None]
    monotone_increasing = all(
        float(reduced[i]["validation_pr_auc"]) <= float(reduced[i + 1]["validation_pr_auc"])
        for i in range(len(reduced) - 1)
    )
    verdict: Dict[str, Any] = {
        "measured": True,
        "n_rungs": len(scored),
        "best_k": best["k"],
        "best_validation_pr_auc": round(float(best["validation_pr_auc"]), 6),
        "range_across_ladder": round(max(values) - min(values), 6),
        "monotone_increasing_in_k_over_reduced_rungs": bool(monotone_increasing),
    }
    for k in (8, 16):
        if k not in by_k:
            verdict[f"k{k}"] = {"present": False}
            continue
        others = {kk: v for kk, v in by_k.items() if kk != k}
        better = sorted([kk for kk, v in others.items() if v > by_k[k]], key=lambda x: (x is None, x))
        verdict[f"k{k}"] = {
            "present": True,
            "validation_pr_auc": round(by_k[k], 6),
            "is_ladder_best": bool(best["k"] == k),
            "n_rungs_strictly_better": len(better),
            "rungs_strictly_better": better,
            "deficit_to_best": round(float(best["validation_pr_auc"]) - by_k[k], 6),
        }
    ci_overlap = True
    best_ci = next(
        (r["bootstrap"] for r in scored if r["k"] == best["k"] and r["bootstrap"].get("measured")),
        None,
    )
    if best_ci is not None:
        overlapping = [
            r["k"]
            for r in scored
            if r["bootstrap"].get("measured")
            and float(r["bootstrap"]["p97_5"]) >= float(best_ci["p2_5"])
        ]
        ci_overlap = len(overlapping) == len(
            [r for r in scored if r["bootstrap"].get("measured")]
        )
        verdict["n_rungs_whose_ci_overlaps_the_best"] = len(overlapping)
    verdict["every_ci_overlaps_the_best"] = bool(ci_overlap)
    verdict["statement"] = (
        "A ladder whose rungs all overlap cannot single out any k. Read with "
        "n_rungs_strictly_better: a k with several rungs above it is not optimal, and a "
        "k that is nominally best with overlapping intervals is not demonstrated optimal "
        "either."
    )
    return verdict


def run(
    *,
    config: Optional[Settings] = None,
    representations: Sequence[str] = tuple(name for name, _ in REPRESENTATIONS),
    components: Sequence[int] = COMPONENTS,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """The full E7 payload."""
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    ladders, context = run_ladder(
        config=config,
        representations=representations,
        components=components,
        draws=draws,
        seed=seed,
    )
    return {
        "ladder_version": LADDER_VERSION,
        "created_at_utc": started.isoformat(),
        "runtime_seconds": round(time.perf_counter() - clock, 1),
        "preregistered": False,
        "what_this_closes": (
            "Evidence sheet §8.4: 'A component sweep for the oral representation is Not "
            "available.' It is now available. §8.5's claim that 16D optimality is "
            "unsupported can be restated from measurement instead of from absence."
        ),
        "condition": PRIMARY,
        "partition_scored": "validation (frozen); TRAIN used for CV and for fitting PCA",
        "arms": list(UNIFORM_ARMS),
        "arm_grid_note": (
            "The E5 uniform grid, so rungs are comparable to each other and to the E5 "
            "lattice. ARM_SPECS is restored on exit; no name is added to E1's frozen "
            "candidate dict."
        ),
        "pca_note": (
            "backend.ml.reduction.DimensionalityReducer -- the deployed reducer -- fitted "
            "on the frozen TRAIN partition only and applied to validation. Not refitted "
            "per CV fold, so the train_cv_pr_auc column is mildly optimistic relative to "
            "E2/E3's fold-internal PCA; the validation column is not affected because "
            "validation rows never enter the fit."
        ),
        "bootstrap": {"draws": int(draws), "seed": int(seed), "unit": "patient"},
        "cross_validation": context["cv_note"],
        "inputs_provenance": context["provenance"],
        "ladders": ladders,
        "test_partition_touched": False,
    }


def _default_output(config: Optional[Settings] = None) -> Path:
    store = ArtifactStore.from_settings(config or default_settings)
    return Path(store.root) / "reports" / "reports_e7_dimension_ladder.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--representation", action="append", dest="representations")
    parser.add_argument("--components", type=int, action="append", dest="components")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    payload = run(
        representations=tuple(args.representations)
        if args.representations
        else tuple(name for name, _ in REPRESENTATIONS),
        components=tuple(args.components) if args.components else COMPONENTS,
        draws=int(args.draws),
        seed=int(args.seed),
    )
    out = args.output or _default_output()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
