"""E5 -- POST-HOC feature-block ablation lattice and split-sensitivity study.

**THIS STUDY IS NOT PRE-REGISTERED.** E1's candidate set (``C1``..``C7``) was frozen
before the first fit and ``candidate_matrices`` forbids additions after the fact. Nothing
here is inserted into that dict, and nothing here may be reported as part of E1. Every
number this module emits carries ``preregistered: false`` and lives in its own artifact
with its own multiplicity accounting, for exactly that reason.

Why it exists
-------------
E1 measured seven representations, but the seven were not a *lattice*: of the three
feature blocks that make up the C7 fusion -- handcrafted CIELAB descriptor (H, 163 dims),
multiscale grid (M, 456 dims) and frozen MobileNetV3-Small embedding (C, 576 dims) --
E1 happened to measure ``{H}`` (C1), ``{C}`` (C6), ``{H,M}`` (C5) and ``{C,H,M}`` (C7)
but never ``{M}``, ``{C,H}`` or ``{C,M}``. So "which block carries the signal" was not
answerable: C7's margin over C6 could have come from H, from M, or only from the pair.

Two further problems E1 left open, both of which this module measures:

* E1's arm grids differ per candidate (C1 got logistic/forest/boosting, C6 got
  linear-SVM/RBF-SVM/forest). A lattice whose cells see different classifier grids
  confounds "this block is better" with "this block got a better search". **Study A**
  therefore re-runs all seven subsets under one identical arm grid.
* E1 reported a single validation split. A margin of 0.913 against 0.889 on 52 images
  is not obviously larger than split noise. **Study B** re-splits the pooled
  TRAIN+VALIDATION patients under many seeds and reports the distribution.

The frozen TEST partition is never read. ``build_inputs`` has no way to name it, and the
pooling in Study B draws only on ``train`` and ``validation``.

Honesty constraints
-------------------
* Study A's cells are *selected from results* in the same sense C7 was, so the lattice is
  reported with patient-level bootstrap intervals, not as a ranking of point estimates.
* Study B's spread is a property of this dataset's 52-image validation partition. A wide
  spread is the finding, not a failure, and is reported either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from backend.core.config import Settings, settings as default_settings
# Reused, never reimplemented: Study A is only comparable to E1 if the standardisation,
# the fold assignment, the hyperparameter selection and the metric call are the *same*
# code objects E1 used. Anything reimplemented here would make a difference between an
# E1 number and an E5 number ambiguous between "the blocks differ" and "the harness does".
from backend.evaluation.e0_controls import N_FOLDS, grouped_folds
from backend.evaluation.e1_failure_map import (
    ARM_SPECS,
    BOOTSTRAP_DRAWS,
    CONDITION_ORACLE_ALL,
    CONDITION_ORACLE_LESION,
    PRIMARY,
    SEED,
    E1Inputs,
    FailureMapError,
    bootstrap_pr_auc,
    build_inputs,
    run_candidate,
)
from backend.ml.artifacts import ArtifactStore

logger = logging.getLogger(__name__)

ABLATION_VERSION = "v1-e5-posthoc-ablation-1"

#: One identical arm grid for every lattice cell. Chosen to be C7's own grid -- the
#: fusion is the thing being ablated, so its classifier family set is the neutral choice.
#: One linear arm, one kernel arm, one ensemble arm: enough to keep a block from losing
#: only because its signal is in a shape one family cannot see.
UNIFORM_ARMS: Tuple[str, ...] = ("logistic", "rbf_svm", "random_forest")

#: The three feature blocks. ``descriptor`` is the raw cached 163-dim CIELAB descriptor
#: (``transform=False``, so not the PCA-reduced quantum vector); ``multiscale`` and
#: ``mobilenet`` are families inside the E1 feature cache.
BLOCKS: Tuple[str, ...] = ("H", "M", "C")
BLOCK_LABELS: Dict[str, str] = {
    "H": "handcrafted CIELAB descriptor (163)",
    "M": "8x8 multiscale grid + centre-vs-surround (456)",
    "C": "frozen MobileNetV3-Small embedding (576)",
}

#: All 2^3 - 1 non-empty subsets, in ascending size then alphabetical order. The four
#: marked ``e1_equivalent`` were measured by E1 under a *different* arm grid; the three
#: marked ``None`` are the cells E1 never measured at all.
LATTICE: Tuple[Tuple[str, Tuple[str, ...], Optional[str]], ...] = (
    ("L_H", ("H",), "C1"),
    ("L_M", ("M",), None),
    ("L_C", ("C",), "C6"),
    ("L_HM", ("H", "M"), "C5"),
    ("L_CH", ("C", "H"), None),
    ("L_CM", ("C", "M"), None),
    ("L_CHM", ("C", "H", "M"), "C7"),
)

#: Resplit seeds for Study B. Seed 42 is included so the frozen split's own fold seed is
#: represented, but note that a seed here re-partitions *patients*, so seed 42 here is a
#: different partition from the frozen manifest's seed-42 partition.
DEFAULT_RESPLIT_SEEDS: Tuple[int, ...] = (
    11, 23, 42, 57, 64, 71, 88, 95, 103, 117,
)


class AblationError(RuntimeError):
    """Raised when the post-hoc study cannot be run as specified."""


# ------------------------------------------------------------------- arm-spec injection
@contextmanager
def _registered_arms(names: Sequence[str], arms: Sequence[str]) -> Iterator[None]:
    """Temporarily teach ``ARM_SPECS`` about the lattice cells, then remove them.

    ``run_candidate`` resolves its arm list through ``ARM_SPECS[name]``, so a new cell
    must be registered to be runnable. The registration is undone on exit: E1's
    ``frozen_grid`` block serialises ``ARM_SPECS`` wholesale, and a leftover ``L_*`` key
    would silently appear in a later E1 artifact as though it had been pre-registered.
    """
    clashes = [name for name in names if name in ARM_SPECS]
    if clashes:
        raise AblationError(
            f"Refusing to shadow pre-registered arm specs {clashes!r}. The post-hoc "
            "lattice must use names disjoint from E1's frozen candidate set."
        )
    for name in names:
        ARM_SPECS[name] = tuple(arms)
    try:
        yield
    finally:
        for name in names:
            ARM_SPECS.pop(name, None)


# ------------------------------------------------------------------------ block plumbing
def _block_matrices(inputs: E1Inputs) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The three feature blocks as ``(train, validation)`` pairs.

    Built from exactly the same attributes ``candidate_matrices`` uses for C1/C5/C6, so a
    lattice cell's columns are byte-identical to the corresponding E1 candidate's columns.
    """
    return {
        "H": (inputs.descriptor_train, inputs.descriptor_validation),
        "M": (inputs.e1_train["multiscale"], inputs.e1_validation["multiscale"]),
        "C": (inputs.e1_train["mobilenet"], inputs.e1_validation["mobilenet"]),
    }


def _lattice_spec(
    subset: Sequence[str],
    blocks: Dict[str, Tuple[np.ndarray, np.ndarray]],
    e1_equivalent: Optional[str],
) -> Dict[str, Any]:
    """One lattice cell's spec in the shape ``run_candidate`` expects.

    Concatenation order follows ``BLOCKS`` (H, M, C) for every cell so that a cell's
    column order is a deterministic function of its subset, not of call order. E1's C5
    used ``[descriptor, multiscale]`` and C7 used ``[mobilenet, multiscale, descriptor]``
    via its own fusion order; column *order* is irrelevant to all three arms here
    (logistic, RBF-SVM and forest are permutation-invariant in their columns once
    standardised), so this does not break comparability with E1.
    """
    ordered = [block for block in BLOCKS if block in subset]
    train = np.hstack([blocks[block][0] for block in ordered])
    validation = np.hstack([blocks[block][1] for block in ordered])
    label = " + ".join(BLOCK_LABELS[block] for block in ordered)
    return {
        "label": f"POST-HOC lattice cell {{{','.join(ordered)}}}: {label}",
        "gap_tested": None,
        "train": train,
        "validation": validation,
        "_subset": ordered,
        "_e1_equivalent": e1_equivalent,
    }


# --------------------------------------------------------------------------- study A
def run_lattice(
    inputs: E1Inputs,
    *,
    seed: int = SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> Dict[str, Any]:
    """Study A: all seven feature-block subsets under one identical arm grid.

    Each cell gets a patient-level bootstrap interval on its best arm's validation
    PR-AUC. The intervals, not the point estimates, are what may be compared: with 52
    validation images the point estimates are far closer together than their own spread.
    """
    folds, cv_note = grouped_folds(
        inputs.y_train, inputs.patients_train, seed=seed, n_splits=N_FOLDS
    )
    blocks = _block_matrices(inputs)
    names = [name for name, _, _ in LATTICE]

    cells: Dict[str, Any] = {}
    with _registered_arms(names, UNIFORM_ARMS):
        for name, subset, equivalent in LATTICE:
            spec = _lattice_spec(subset, blocks, equivalent)
            logger.info("  %s (%d dims): %s", name, spec["train"].shape[1], "/".join(subset))
            payload = run_candidate(name, spec, inputs, folds, seed=seed)
            internal = payload.pop("_arms_internal")

            best = payload["best_arm"]
            if best is not None:
                scores = internal[best]["_validation_probability"]
                if scores is None:
                    scores = internal[best]["_validation_score"]
                payload["bootstrap"] = bootstrap_pr_auc(
                    inputs.y_validation,
                    np.asarray(scores, dtype=np.float64),
                    inputs.patients_validation,
                    draws=bootstrap_draws,
                    seed=seed,
                )
            else:
                payload["bootstrap"] = None

            payload["subset"] = list(subset)
            payload["e1_equivalent"] = equivalent
            payload["preregistered"] = False
            cells[name] = payload

    return {
        "preregistered": False,
        "arm_grid": list(UNIFORM_ARMS),
        "arm_grid_note": (
            "identical for every cell, so a cell's score cannot differ from another's "
            "because it was given a different classifier search"
        ),
        "cv_note": cv_note,
        "n_cells": len(cells),
        "n_tests": len(cells) * len(UNIFORM_ARMS),
        "multiplicity_note": (
            f"{len(cells)} cells x {len(UNIFORM_ARMS)} arms = {len(cells) * len(UNIFORM_ARMS)} "
            "fits on one condition, chosen AFTER E1's results were seen. No p-value is "
            "computed here and none may be quoted from this study; the bootstrap "
            "intervals are descriptive."
        ),
        "cells": cells,
        "marginal_contributions": _marginal_contributions(cells),
    }


def _marginal_contributions(cells: Dict[str, Any]) -> Dict[str, Any]:
    """Every block's marginal PR-AUC delta over every subset it can be added to.

    This is the question E1 could not answer. For block B and subset S not containing B,
    the marginal is ``score(S + B) - score(S)``. Averaging those deltas over all S gives
    each block a Shapley-style contribution that does not depend on an arbitrary order
    of addition -- with three blocks the exact Shapley value is cheap, so it is computed
    rather than approximated.
    """
    by_subset: Dict[frozenset, Optional[float]] = {}
    for name, subset, _ in LATTICE:
        payload = cells.get(name)
        by_subset[frozenset(subset)] = (
            None if payload is None else payload["best_validation_pr_auc"]
        )
    by_subset[frozenset()] = None  # empty model was never fitted; marginals skip it

    marginals: Dict[str, List[Dict[str, Any]]] = {block: [] for block in BLOCKS}
    for block in BLOCKS:
        for subset, score in list(by_subset.items()):
            if block in subset or not subset:
                continue
            grown = by_subset.get(subset | {block})
            if score is None or grown is None:
                continue
            marginals[block].append(
                {
                    "over": "".join(sorted(subset)),
                    "from": round(score, 6),
                    "to": round(grown, 6),
                    "delta": round(grown - score, 6),
                }
            )

    shapley: Dict[str, Optional[float]] = {}
    for block in BLOCKS:
        deltas = [entry["delta"] for entry in marginals[block]]
        shapley[block] = round(float(np.mean(deltas)), 6) if deltas else None

    return {
        "per_block": marginals,
        "mean_marginal_delta": shapley,
        "note": (
            "mean over the non-empty subsets each block can be added to. The singleton "
            "model (block alone vs nothing) is excluded because no empty-feature model "
            "was fitted, so these are means over 2 of the 3 Shapley coalitions and are "
            "NOT exact Shapley values."
        ),
    }


# --------------------------------------------------------------------------- study B
def _patient_resplit(
    patients: Sequence[str],
    labels: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-partition rows at the patient level, stratified on patient positivity.

    Mirrors ``build_patient_level_split``'s two mechanisms deliberately: SHA-256 hash
    ordering keyed on ``(seed, patient_id)`` rather than ``random.shuffle``, and
    largest-remainder allocation within each stratum. A patient contributes all of its
    images to exactly one side, which is the invariant the whole project rests on.
    """
    labels = np.asarray(labels, dtype=int)
    rows_by_patient: Dict[str, List[int]] = {}
    for position, patient in enumerate(patients):
        rows_by_patient.setdefault(patient, []).append(position)

    strata: Dict[int, List[str]] = {0: [], 1: []}
    for patient, rows in rows_by_patient.items():
        strata[int(labels[rows].max())].append(patient)

    train_patients: List[str] = []
    validation_patients: List[str] = []
    for stratum in (1, 0):  # positives first: the scarce stratum sets the rounding
        ordered = sorted(
            strata[stratum],
            key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode("utf-8")).hexdigest(),
        )
        n = len(ordered)
        exact = n * train_fraction
        n_train = int(exact)
        if (exact - n_train) >= 0.5 and n_train < n:
            n_train += 1
        n_train = max(1, min(n - 1, n_train)) if n >= 2 else n
        train_patients.extend(ordered[:n_train])
        validation_patients.extend(ordered[n_train:])

    train_rows = np.asarray(
        sorted(r for p in train_patients for r in rows_by_patient[p]), dtype=int
    )
    validation_rows = np.asarray(
        sorted(r for p in validation_patients for r in rows_by_patient[p]), dtype=int
    )
    overlap = set(train_patients) & set(validation_patients)
    if overlap:
        raise AblationError(f"Resplit leaked {len(overlap)} patients across partitions.")
    return train_rows, validation_rows


def _pool(inputs: E1Inputs) -> Dict[str, Any]:
    """Concatenate TRAIN and VALIDATION into one pool. TEST is never involved.

    The pool is what gets re-split. Pooling is sound here because the two partitions were
    themselves drawn patient-disjointly from the same cohort, so no patient appears twice
    in the concatenation -- asserted below rather than assumed.
    """
    patients = list(inputs.patients_train) + list(inputs.patients_validation)
    if set(inputs.patients_train) & set(inputs.patients_validation):
        raise AblationError(
            "TRAIN and VALIDATION share patients in the frozen manifest; pooling them "
            "would create a leaked pool."
        )
    return {
        "patients": patients,
        "ids": list(inputs.ids_train) + list(inputs.ids_validation),
        "y": np.concatenate([inputs.y_train, inputs.y_validation]),
        "descriptor": np.vstack([inputs.descriptor_train, inputs.descriptor_validation]),
        "multiscale": np.vstack(
            [inputs.e1_train["multiscale"], inputs.e1_validation["multiscale"]]
        ),
        "mobilenet": np.vstack(
            [inputs.e1_train["mobilenet"], inputs.e1_validation["mobilenet"]]
        ),
        "n_train_original": int(inputs.y_train.size),
        "n_validation_original": int(inputs.y_validation.size),
    }


def _inputs_from_pool(
    pool: Dict[str, Any], train_rows: np.ndarray, validation_rows: np.ndarray, condition: str
) -> E1Inputs:
    """Wrap a resplit of the pool back into an ``E1Inputs`` so ``run_candidate`` works.

    ``pixels_*`` are given zero-width placeholders: no lattice cell reads them, and
    carrying the real 126 MB pixel cache through ten resplits would cost memory for
    columns nothing in this study touches. Any attempt to build a pixel candidate from
    these inputs would fail loudly on a zero-column hstack rather than fit silently.
    """
    empty_train = np.empty((train_rows.size, 0), dtype=np.uint8)
    empty_validation = np.empty((validation_rows.size, 0), dtype=np.uint8)
    return E1Inputs(
        condition=condition,
        descriptor_train=pool["descriptor"][train_rows],
        descriptor_validation=pool["descriptor"][validation_rows],
        pixels_train=empty_train,
        pixels_validation=empty_validation,
        e1_train={
            "multiscale": pool["multiscale"][train_rows],
            "mobilenet": pool["mobilenet"][train_rows],
        },
        e1_validation={
            "multiscale": pool["multiscale"][validation_rows],
            "mobilenet": pool["mobilenet"][validation_rows],
        },
        y_train=pool["y"][train_rows],
        y_validation=pool["y"][validation_rows],
        patients_train=[pool["patients"][i] for i in train_rows],
        patients_validation=[pool["patients"][i] for i in validation_rows],
        ids_train=[pool["ids"][i] for i in train_rows],
        ids_validation=[pool["ids"][i] for i in validation_rows],
    )


def run_split_sensitivity(
    inputs: E1Inputs,
    *,
    seeds: Sequence[int] = DEFAULT_RESPLIT_SEEDS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Study B: the lattice re-fitted under many patient-level re-splits.

    Answers the question E1's single split cannot: is the C7-over-C6 margin a property of
    the representation, or of which 52 images happened to land in validation? Reports the
    full per-seed distribution and how often each ordering holds -- not a mean alone,
    because a mean would hide a bimodal spread.
    """
    pool = _pool(inputs)
    train_fraction = pool["n_train_original"] / (
        pool["n_train_original"] + pool["n_validation_original"]
    )
    names = [name for name, _, _ in LATTICE]

    per_seed: List[Dict[str, Any]] = []
    with _registered_arms(names, UNIFORM_ARMS):
        for resplit_seed in seeds:
            train_rows, validation_rows = _patient_resplit(
                pool["patients"], pool["y"], seed=resplit_seed, train_fraction=train_fraction
            )
            resplit = _inputs_from_pool(pool, train_rows, validation_rows, inputs.condition)
            if resplit.y_validation.sum() < 2 or resplit.y_train.sum() < 2:
                per_seed.append(
                    {
                        "resplit_seed": resplit_seed,
                        "skipped": "fewer than 2 positives on one side of the resplit",
                        "shape": resplit.shape_note,
                    }
                )
                continue

            folds, _ = grouped_folds(
                resplit.y_train, resplit.patients_train, seed=seed, n_splits=N_FOLDS
            )
            blocks = _block_matrices(resplit)
            scores: Dict[str, Optional[float]] = {}
            best_arms: Dict[str, Optional[str]] = {}
            for name, subset, equivalent in LATTICE:
                spec = _lattice_spec(subset, blocks, equivalent)
                payload = run_candidate(name, spec, resplit, folds, seed=seed)
                payload.pop("_arms_internal", None)
                scores[name] = payload["best_validation_pr_auc"]
                best_arms[name] = payload["best_arm"]
            logger.info(
                "  resplit seed %d: %d/%d rows, L_CHM=%s",
                resplit_seed,
                resplit.y_train.size,
                resplit.y_validation.size,
                "n/a" if scores["L_CHM"] is None else f"{scores['L_CHM']:.4f}",
            )
            per_seed.append(
                {
                    "resplit_seed": resplit_seed,
                    "shape": resplit.shape_note,
                    "best_validation_pr_auc": {
                        k: (None if v is None else round(v, 6)) for k, v in scores.items()
                    },
                    "best_arm": best_arms,
                }
            )

    return {
        "preregistered": False,
        "n_seeds_requested": len(seeds),
        "n_seeds_completed": sum(1 for e in per_seed if "skipped" not in e),
        "resplit_seeds": list(seeds),
        "train_fraction": round(train_fraction, 6),
        "pool": {
            "n_rows": int(pool["y"].size),
            "n_positive": int(pool["y"].sum()),
            "n_patients": len(set(pool["patients"])),
            "source": "frozen TRAIN + frozen VALIDATION of the primary condition",
            "test_partition_touched": False,
        },
        "per_seed": per_seed,
        "summary": _sensitivity_summary(per_seed),
        "interpretation_note": (
            "A resplit changes which patients are scored, so the spread here is the "
            "sampling variability of the validation estimate itself. It is NOT a "
            "confidence interval for the frozen split's number and must not be quoted "
            "as one."
        ),
    }


def _sensitivity_summary(per_seed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-cell distribution across resplits, plus how stable the orderings are."""
    completed = [entry for entry in per_seed if "skipped" not in entry]
    stats: Dict[str, Any] = {}
    for name, _, equivalent in LATTICE:
        values = [
            entry["best_validation_pr_auc"][name]
            for entry in completed
            if entry["best_validation_pr_auc"].get(name) is not None
        ]
        if not values:
            stats[name] = None
            continue
        array = np.asarray(values, dtype=np.float64)
        stats[name] = {
            "e1_equivalent": equivalent,
            "n": int(array.size),
            "mean": round(float(array.mean()), 6),
            "sd": round(float(array.std(ddof=1)), 6) if array.size > 1 else None,
            "min": round(float(array.min()), 6),
            "max": round(float(array.max()), 6),
            "range": round(float(array.max() - array.min()), 6),
        }

    orderings: Dict[str, Any] = {}
    for label, better, worse in (
        ("L_CHM_beats_L_C", "L_CHM", "L_C"),
        ("L_CHM_beats_L_H", "L_CHM", "L_H"),
        ("L_C_beats_L_H", "L_C", "L_H"),
        ("L_CHM_is_best_cell", "L_CHM", None),
    ):
        wins = 0
        total = 0
        for entry in completed:
            scores = entry["best_validation_pr_auc"]
            if worse is None:
                values = {k: v for k, v in scores.items() if v is not None}
                if len(values) < len(LATTICE):
                    continue
                total += 1
                wins += int(max(values, key=lambda k: values[k]) == better)
            else:
                if scores.get(better) is None or scores.get(worse) is None:
                    continue
                total += 1
                wins += int(scores[better] > scores[worse])
        orderings[label] = {
            "wins": wins,
            "of": total,
            "rate": round(wins / total, 4) if total else None,
        }

    return {"per_cell": stats, "ordering_stability": orderings}


# ------------------------------------------------------------------------------ driver
def run(
    *,
    config: Optional[Settings] = None,
    condition: str = PRIMARY,
    resplit_seeds: Sequence[int] = DEFAULT_RESPLIT_SEEDS,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
    skip_sensitivity: bool = False,
) -> Dict[str, Any]:
    """Run both post-hoc studies on one condition and return the full payload."""
    started = time.perf_counter()
    if condition not in (CONDITION_ORACLE_LESION, CONDITION_ORACLE_ALL):
        raise AblationError(
            f"Condition {condition!r} is not one of the two oracle conditions E1 measured."
        )

    per_condition, provenance = build_inputs(config=config, conditions=(condition,))
    inputs = per_condition[condition]
    logger.info(
        "E5 post-hoc on %s: %d train / %d validation rows (%d / %d positive)",
        condition,
        inputs.y_train.size,
        inputs.y_validation.size,
        int(inputs.y_train.sum()),
        int(inputs.y_validation.sum()),
    )

    logger.info("Study A: feature-block lattice (%d cells)", len(LATTICE))
    lattice = run_lattice(inputs, seed=seed, bootstrap_draws=bootstrap_draws)

    if skip_sensitivity:
        sensitivity: Dict[str, Any] = {
            "skipped": "requested with --skip-sensitivity",
            "preregistered": False,
        }
    else:
        logger.info("Study B: split sensitivity (%d resplits)", len(resplit_seeds))
        sensitivity = run_split_sensitivity(inputs, seeds=resplit_seeds, seed=seed)

    return {
        "version": ABLATION_VERSION,
        "preregistered": False,
        "status_note": (
            "POST-HOC. E1's candidate set was frozen before its first fit and nothing "
            "here was added to it. These numbers extend E1's ablation coverage and must "
            "be reported as a separate, explicitly post-hoc study with their own "
            "multiplicity accounting. No p-value is computed and none may be quoted."
        ),
        "condition": condition,
        "shape": inputs.shape_note,
        "blocks": BLOCK_LABELS,
        "provenance": provenance,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "seed": seed,
        "study_a_lattice": lattice,
        "study_b_split_sensitivity": sensitivity,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def _default_output(config: Optional[Settings] = None) -> Path:
    """Alongside the E4 reports rather than at the repo root where E1 landed.

    ``ArtifactStore`` has no ``reports_dir``, so the directory is derived from its root
    the same way the E4 scripts address it.
    """
    store = ArtifactStore.from_settings(config or default_settings)
    return Path(store.root) / "reports" / "reports_e5_posthoc_ablation.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="E5 POST-HOC feature-block ablation lattice and split sensitivity."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--condition", type=str, default=PRIMARY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--resplit-seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_RESPLIT_SEEDS),
        help="patient-level resplit seeds for Study B",
    )
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        payload = run(
            condition=args.condition,
            resplit_seeds=tuple(args.resplit_seeds),
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed,
            skip_sensitivity=args.skip_sensitivity,
        )
    except (AblationError, FailureMapError) as exc:
        logger.error("%s", exc)
        return 1

    output = args.output or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
