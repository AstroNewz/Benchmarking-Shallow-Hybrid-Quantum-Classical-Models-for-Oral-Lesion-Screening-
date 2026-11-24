"""E6 -- patient-level bootstrap intervals on the **frozen test partition** (POST-HOC).

Every test-partition number this project reports is a bare point estimate resting on
18 positives. That is the single largest statistical gap in the evidence sheet: a
reader cannot tell 0.92562 from 0.75 without an interval, and at this sample size the
difference between those two is a handful of patients.

**Nothing is selected here.** The models are the three frozen baseline records,
refitted on TRAIN exactly as :func:`pixel_final_test.classical_test_scores` does; the
operating threshold is the one ``baselines.json`` recorded from validation. This module
adds *uncertainty* to numbers that were already computed and already reported -- it
does not compute a new number, choose a model, move a threshold, or compare arms.
``selection_performed: false`` is asserted in the artifact and is the whole point: a
confidence interval around a frozen estimate is not a second look at the test set in
the sense that matters, because no decision is conditioned on it.

Two quantities per condition:

1. **Marginal intervals** for ROC-AUC, PR-AUC, sensitivity and specificity, by
   resampling *patients* with replacement (images from one patient share a diagnosis;
   an image-level bootstrap would understate the interval, which is the direction that
   manufactures significance -- the same argument as E1 §9.4).

2. **The paired model-minus-leak-bar delta.** DEC-024 established that ROI selection
   which consults annotations encodes the label in its crop geometry, and that the
   honest comparison floor for a condition-A metric is ``area_only_roc_auc``, not 0.5.
   That floor has never had an interval either, and -- more importantly -- the
   *difference* has never been tested. Both quantities are recomputed on the **same
   resampled rows within each draw**, so the delta is genuinely paired and its interval
   answers the question the leak bar was invented to ask: does the classifier beat the
   geometry, or is it reading it?

The leak-bar score is ``-roi_area_fraction``. ``state_diagnostics.roi_geometry_leak``
rank-transforms it to the unit interval first; ROC-AUC and average precision are both
rank statistics and therefore invariant under that monotone map, so the point estimates
reproduce exactly. The module asserts this against the frozen artifact rather than
assuming it.

Run::

    python -m backend.evaluation.e6_test_intervals

POST-HOC. Not pre-registered. No p-value is reported and no hypothesis is tested; the
intervals are descriptive, and the one delta that excludes zero is labelled as a
post-hoc observation, not a confirmation.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
from backend.evaluation.e1_failure_map import _average_precision
from backend.evaluation.metrics import evaluate_predictions
from backend.evaluation.pixel_final_test import classical_test_scores
from backend.evaluation.state_diagnostics import roi_geometry_leak
from backend.ml.artifacts import ArtifactStore
from backend.training.pixel_data import (
    CONDITION_DESCRIPTIONS,
    PixelPartition,
    condition_rows,
    load_pixel_partitions,
)

logger = logging.getLogger(__name__)

INTERVAL_VERSION = "v1-e6-test-intervals-1"
BOOTSTRAP_DRAWS = 2000
SEED = 42
ROI_MODES: Tuple[str, ...] = ("oracle", "predicted")
METRIC_KEYS: Tuple[str, ...] = ("roc_auc", "pr_auc", "sensitivity", "specificity")
# The point estimates in reports_final_{oracle,predicted}.json are rounded to 6 dp;
# anything inside half a unit in the last place is a reproduction, not a discrepancy.
ANCHOR_TOLERANCE = 5e-7


class IntervalError(RuntimeError):
    """Raised when the recomputation fails to reproduce a frozen point estimate."""


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Mann-Whitney ROC-AUC with midrank ties. ``None`` when a class is absent."""
    y = np.asarray(y, dtype=int)
    positive = int(y.sum())
    negative = int(y.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or values[index] != values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def _rate_pair(y: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    """``(sensitivity, specificity)`` at a fixed threshold; 0.0 for empty denominators."""
    y = np.asarray(y, dtype=int)
    flag = np.asarray(scores, dtype=np.float64) >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    fp = int(np.sum(flag & (y == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity, specificity


def _patient_blocks(patients: Sequence[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Row positions grouped by patient, preserving first-appearance order."""
    order: Dict[str, List[int]] = {}
    for position, patient in enumerate(patients):
        order.setdefault(str(patient), []).append(position)
    return list(order), {k: np.asarray(v, dtype=int) for k, v in order.items()}


def _interval(values: Sequence[float], *, draws: int, label: str) -> Dict[str, Any]:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size < 2:
        return {
            "measured": False,
            "reason": "fewer than two usable bootstrap draws",
            "n_draws_requested": int(draws),
            "quantity": label,
        }
    return {
        "measured": True,
        "quantity": label,
        "n_draws_requested": int(draws),
        "n_draws_usable": int(array.size),
        "unit": "patient (resampled with replacement)",
        "mean": round(float(array.mean()), 6),
        "standard_error": round(float(array.std(ddof=1)), 6),
        "p2_5": round(float(np.percentile(array, 2.5)), 6),
        "p50": round(float(np.percentile(array, 50.0)), 6),
        "p97_5": round(float(np.percentile(array, 97.5)), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
    }


def _delta_block(values: Sequence[float], *, draws: int, label: str) -> Dict[str, Any]:
    """A paired-difference interval, with the sign test the interval implies."""
    block = _interval(values, draws=draws, label=label)
    if not block.get("measured"):
        return block
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    block["excludes_zero"] = bool(block["p2_5"] > 0.0 or block["p97_5"] < 0.0)
    block["fraction_of_draws_above_zero"] = round(float(np.mean(array > 0.0)), 6)
    block["interpretation"] = (
        "Positive means the classifier ranked these rows better than ROI area alone on "
        "the same resampled patients. An interval containing zero means the condition's "
        "headline discrimination is not separable from its crop geometry at this sample "
        "size. POST-HOC: descriptive, not a pre-registered test."
    )
    return block


def _area_scores(partition: PixelPartition, rows: np.ndarray) -> np.ndarray:
    """The leak-bar score: tighter crop -> more suspicious (DEC-024's direction)."""
    return -np.asarray(partition.roi_area_fraction, dtype=np.float64)[rows]


def _condition_block(
    *,
    partition: PixelPartition,
    rows: np.ndarray,
    condition: str,
    scores_by_name: Dict[str, Dict[str, float]],
    threshold: float,
    frozen: Optional[Dict[str, Any]],
    draws: int,
    seed: int,
) -> Dict[str, Any]:
    """Marginal and paired intervals for every baseline on one condition."""
    labels = np.asarray(partition.labels, dtype=int)[rows]
    image_ids = [partition.image_ids[int(r)] for r in rows]
    patients = [str(partition.patient_ids[int(r)]) for r in rows]
    area = _area_scores(partition, rows)
    finite = np.isfinite(area)

    leak = roi_geometry_leak(partition, rows)
    entry: Dict[str, Any] = {
        "condition": condition,
        "description": CONDITION_DESCRIPTIONS.get(condition, ""),
        "partition": "test",
        "n_samples": int(rows.size),
        "n_positive": int(labels.sum()),
        "n_negative": int(rows.size - int(labels.sum())),
        "n_patients": len(set(patients)),
        "n_patients_with_a_positive": len(
            {p for p, y in zip(patients, labels.tolist()) if y == 1}
        ),
        "prevalence": round(float(labels.mean()), 6) if rows.size else None,
        "leak_bar_point": {
            "area_only_roc_auc": leak.get("area_only_roc_auc"),
            "area_only_pr_auc": leak.get("area_only_pr_auc"),
            "n_finite_area": int(finite.sum()),
        },
        "baselines": {},
    }

    # Mirror pixel_final_test._metrics exactly: a single-class subset has no ROC-AUC,
    # no average precision and no meaningful sensitivity, and the frozen artifact
    # records ``metrics: null`` for it. Bootstrapping such a condition would produce
    # intervals around quantities that do not exist.
    if rows.size == 0 or len(set(labels.tolist())) < 2:
        entry["measurable"] = False
        entry["reason"] = (
            "the condition contains a single class on the test partition; the frozen "
            "artifact reports metrics: null here and this module reports no interval"
        )
        return entry
    entry["measurable"] = True

    unique, blocks = _patient_blocks(patients)
    rng = np.random.default_rng(seed)
    # One resample sequence, shared by every baseline and by the leak bar, so that all
    # deltas within this condition are paired on identical rows.
    draw_rows: List[np.ndarray] = []
    for _ in range(int(draws)):
        picked: List[int] = []
        for patient in rng.choice(unique, size=len(unique), replace=True):
            picked.extend(blocks[str(patient)].tolist())
        draw_rows.append(np.asarray(picked, dtype=int))

    leak_roc: List[Optional[float]] = []
    leak_pr: List[Optional[float]] = []
    for sel in draw_rows:
        mask = finite[sel]
        leak_roc.append(_roc_auc(labels[sel][mask], area[sel][mask]))
        leak_pr.append(_average_precision(labels[sel][mask], area[sel][mask]))
    entry["leak_bar_interval"] = {
        "roc_auc": _interval(
            [v for v in leak_roc if v is not None], draws=draws, label="area_only_roc_auc"
        ),
        "pr_auc": _interval(
            [v for v in leak_pr if v is not None], draws=draws, label="area_only_pr_auc"
        ),
    }

    for name, by_id in sorted(scores_by_name.items()):
        mask = np.asarray([i in by_id for i in image_ids], dtype=bool)
        if not mask.any():
            entry["baselines"][name] = {
                "measured": False,
                "reason": "no row of this condition is in the descriptor cache",
            }
            continue
        y = labels[mask]
        s = np.asarray([by_id[i] for i, keep in zip(image_ids, mask) if keep], dtype=np.float64)
        point = evaluate_predictions(y, s, threshold=threshold)
        sens, spec = _rate_pair(y, s, threshold)

        # Re-index the shared draws into this baseline's scored subset so the pairing
        # with the leak bar survives rows dropped for a missing descriptor.
        position = -np.ones(rows.size, dtype=int)
        position[np.flatnonzero(mask)] = np.arange(int(mask.sum()))
        collected: Dict[str, List[Optional[float]]] = {k: [] for k in METRIC_KEYS}
        deltas: List[Optional[float]] = []
        for sel, bar in zip(draw_rows, leak_roc):
            local = position[sel]
            local = local[local >= 0]
            if local.size < 2:
                continue
            yb, sb = y[local], s[local]
            roc = _roc_auc(yb, sb)
            collected["roc_auc"].append(roc)
            collected["pr_auc"].append(_average_precision(yb, sb))
            sb_sens, sb_spec = _rate_pair(yb, sb, threshold)
            collected["sensitivity"].append(sb_sens)
            collected["specificity"].append(sb_spec)
            if roc is not None and bar is not None:
                deltas.append(roc - bar)

        block: Dict[str, Any] = {
            "measured": True,
            "n_scored": int(mask.sum()),
            "n_not_in_descriptor_cache": int(rows.size - int(mask.sum())),
            "threshold": float(threshold),
            "threshold_source": "baselines.json:screening_threshold (selected on validation)",
            "point": {
                "roc_auc": point.roc_auc,
                "pr_auc": point.pr_auc,
                "sensitivity": round(float(sens), 6),
                "specificity": round(float(spec), 6),
            },
            "interval": {
                key: _interval(
                    [v for v in collected[key] if v is not None], draws=draws, label=key
                )
                for key in METRIC_KEYS
            },
            "roc_auc_minus_leak_bar": _delta_block(
                deltas, draws=draws, label="roc_auc - area_only_roc_auc (paired)"
            ),
        }

        if frozen is not None:
            stored = (frozen.get("classical", {}).get(name, {}) or {}).get("metrics") or {}
            if not stored:
                block["anchor_check"] = {
                    "frozen_metrics_absent": True,
                    "note": (
                        "the frozen artifact records no metric block for this "
                        "condition/baseline, so there is nothing to reproduce against"
                    ),
                }
                entry["baselines"][name] = block
                continue
            checks: Dict[str, Any] = {}
            for key in ("roc_auc", "pr_auc", "sensitivity", "specificity"):
                want, got = stored.get(key), block["point"][key]
                if want is None or got is None:
                    checks[key] = {"frozen": want, "recomputed": got, "reproduced": want == got}
                    continue
                checks[key] = {
                    "frozen": want,
                    "recomputed": round(float(got), 6),
                    "reproduced": abs(float(want) - float(got)) <= ANCHOR_TOLERANCE,
                }
            block["anchor_check"] = checks
            failed = [k for k, v in checks.items() if not v["reproduced"]]
            if failed:
                raise IntervalError(
                    f"Recomputed {condition}/{name} does not reproduce the frozen point "
                    f"estimate for {failed!r}: {checks}. Refusing to publish intervals "
                    "around a number this module cannot reproduce."
                )
        entry["baselines"][name] = block

    if frozen is not None:
        want = (frozen.get("roi_geometry_leak") or {}).get("area_only_roc_auc")
        got = entry["leak_bar_point"]["area_only_roc_auc"]
        reproduced = (
            want is None and got is None
        ) or (
            want is not None and got is not None and abs(float(want) - float(got)) <= 1e-12
        )
        entry["leak_bar_anchor_check"] = {
            "frozen": want,
            "recomputed": got,
            "reproduced": bool(reproduced),
        }
        if not reproduced:
            raise IntervalError(
                f"Leak bar for {condition} does not reproduce: frozen {want!r}, "
                f"recomputed {got!r}."
            )
    return entry


def run(
    *,
    config: Optional[Settings] = None,
    roi_modes: Sequence[str] = ROI_MODES,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Bootstrap intervals for every frozen test-partition classical metric."""
    cfg = config or default_settings
    started = datetime.now(timezone.utc)
    classical = classical_test_scores(config=cfg, model_version=model_version)
    threshold = float(classical["threshold"])
    scores_by_name = classical["scores"]

    modes: Dict[str, Any] = {}
    for roi_mode in roi_modes:
        logger.info("E6: %s", roi_mode)
        data = load_pixel_partitions(config=cfg, roi_mode=roi_mode)
        partition = data.test
        frozen_path = Path(f"reports_final_{roi_mode}.json")
        frozen_by_condition: Dict[str, Any] = {}
        if frozen_path.exists():
            payload = json.loads(frozen_path.read_text(encoding="utf-8"))
            frozen_by_condition = {
                str(c.get("condition")): c for c in payload.get("conditions", [])
            }
        conditions = condition_rows(partition)
        blocks = []
        for name, rows in sorted(conditions.items()):
            rows = np.asarray(rows, dtype=int)
            logger.info("  %s: %d rows", name, rows.size)
            blocks.append(
                _condition_block(
                    partition=partition,
                    rows=rows,
                    condition=name,
                    scores_by_name=scores_by_name,
                    threshold=threshold,
                    frozen=frozen_by_condition.get(name),
                    draws=draws,
                    seed=seed,
                )
            )
        modes[roi_mode] = {
            "roi_mode": roi_mode,
            "frozen_artifact": str(frozen_path) if frozen_path.exists() else None,
            "n_test_rows": int(partition.labels.size),
            "conditions": blocks,
        }

    return {
        "interval_version": INTERVAL_VERSION,
        "created_at_utc": started.isoformat(),
        "runtime_seconds": round(
            (datetime.now(timezone.utc) - started).total_seconds(), 1
        ),
        "preregistered": False,
        "selection_performed": False,
        "what_this_is": (
            "Patient-level bootstrap intervals around test-partition point estimates "
            "that were already frozen and already reported. No model, threshold, "
            "representation or condition was chosen here."
        ),
        "what_this_is_not": (
            "Not a new test-set evaluation and not a model comparison. No p-value is "
            "reported. The one interval that excludes zero (the paired model-minus-"
            "leak-bar delta) is a post-hoc observation and must be labelled as such."
        ),
        "bootstrap": {
            "draws": int(draws),
            "seed": int(seed),
            "unit": "patient (resampled with replacement)",
            "shared_draws_within_condition": True,
            "why_patient_level": (
                "Images from one patient share a diagnosis. An image-level bootstrap "
                "treats correlated rows as independent and understates the interval, "
                "which is the direction that manufactures significance."
            ),
        },
        "classical_reference": {
            "model_version": classical["model_version"],
            "threshold": threshold,
            "threshold_source": classical["threshold_source"],
            "records": classical["records"],
            "refit_partition": "train",
        },
        "leak_bar_note": (
            "The leak-bar score is -roi_area_fraction. roi_geometry_leak rank-transforms "
            "it to [0,1] first; ROC-AUC and average precision are rank statistics and "
            "invariant under that monotone map, which the anchor check verifies to 1e-12."
        ),
        "modes": modes,
    }


def _default_output(config: Optional[Settings] = None) -> Path:
    store = ArtifactStore.from_settings(config or default_settings)
    return Path(store.root) / "reports" / "reports_e6_test_intervals.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--roi-mode", action="append", dest="roi_modes")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    payload = run(
        roi_modes=tuple(args.roi_modes) if args.roi_modes else ROI_MODES,
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
