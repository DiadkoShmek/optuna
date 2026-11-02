from __future__ import annotations

import itertools
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

import optuna


def test_wilcoxon_pruner_constructor() -> None:
    optuna.pruners.WilcoxonPruner()
    optuna.pruners.WilcoxonPruner(p_threshold=0)
    optuna.pruners.WilcoxonPruner(p_threshold=1)
    optuna.pruners.WilcoxonPruner(p_threshold=0.05)
    optuna.pruners.WilcoxonPruner(n_startup_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.WilcoxonPruner(p_threshold=-0.1)

    with pytest.raises(ValueError):
        optuna.pruners.WilcoxonPruner(n_startup_steps=-5)


def test_wilcoxon_pruner_first_trial() -> None:
    # A pruner is not activated at a first trial.
    pruner = optuna.pruners.WilcoxonPruner()
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    assert not trial.should_prune()
    trial.report(1, 1)
    assert not trial.should_prune()
    trial.report(2, 2)
    assert not trial.should_prune()


def test_wilcoxon_pruner_when_best_trial_has_no_intermediate_value() -> None:
    # A pruner is not activated at a first trial.
    pruner = optuna.pruners.WilcoxonPruner()
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    study.tell(trial, 10)
    trial = study.ask()
    assert not trial.should_prune()
    trial.report(1, 1)
    assert not trial.should_prune()
    trial.report(2, 2)
    assert not trial.should_prune()


@pytest.mark.parametrize(
    "p_threshold,step_values,expected_should_prune",
    [
        (0.2, [-1, 1, 2, 3, 4, 5, -2, -3], [False, False, False, True, True, True, True, True]),
        (0.15, [-1, 1, 2, 3, 4, 5, -2, -3], [False, False, False, False, True, True, True, False]),
    ],
)
def test_wilcoxon_pruner_normal(
    p_threshold: float,
    step_values: list[float],
    expected_should_prune: list[bool],
) -> None:
    pruner = optuna.pruners.WilcoxonPruner(n_startup_steps=0, p_threshold=p_threshold)
    study = optuna.study.create_study(pruner=pruner)

    # Insert the best trial
    study.add_trial(
        optuna.trial.create_trial(
            value=0,
            params={},
            distributions={},
            intermediate_values={step: 0 for step in range(10)},
        )
    )

    trial = study.ask()

    should_prune = [False] * len(step_values)

    for step_i in range(len(step_values)):
        trial.report(step_values[step_i], step_i)
        should_prune[step_i] = trial.should_prune()

    assert should_prune == expected_should_prune


@pytest.mark.parametrize(
    "best_intermediate_values,intermediate_values",
    [
        ({1: 1}, {1: 1, 2: 2}),  # Current trial has more steps than the best trial
        ({1: 1}, {1: float("nan")}),  # NaN value
        ({1: float("nan")}, {1: 1}),  # NaN value
        ({1: 1}, {1: float("inf")}),  # infinite value
        ({1: float("inf")}, {1: 1}),  # infinite value
    ],
)
@pytest.mark.parametrize(
    "direction",
    ("minimize", "maximize"),
)
def test_wilcoxon_pruner_warn_bad_best_trial(
    best_intermediate_values: dict[int, float],
    intermediate_values: dict[int, float],
    direction: Literal["minimize", "maximize"],
) -> None:
    pruner = optuna.pruners.WilcoxonPruner()
    study = optuna.study.create_study(direction=direction, pruner=pruner)

    # Insert best trial
    study.add_trial(
        optuna.trial.create_trial(
            value=0, params={}, distributions={}, intermediate_values=best_intermediate_values
        )
    )
    trial = study.ask()
    with pytest.warns(UserWarning):
        for step, value in intermediate_values.items():
            trial.report(value, step)
            trial.should_prune()


def test_wilcoxon_pruner_if_average_is_best_then_not_prune() -> None:
    pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.5)
    study = optuna.study.create_study(direction="minimize", pruner=pruner)

    best_intermediate_values_value = [0.0 for _ in range(10)] + [8.0 for _ in range(10)]
    best_intermediate_values = dict(zip(list(range(20)), best_intermediate_values_value))

    # Insert best trial
    study.add_trial(
        optuna.trial.create_trial(
            value=4.0, params={}, distributions={}, intermediate_values=best_intermediate_values
        )
    )
    trial = study.ask()
    intermediate_values = [1.0 for _ in range(10)] + [9.0 for _ in range(10)]
    for step, value in enumerate(intermediate_values):
        trial.report(value, step)
        average = sum(intermediate_values[: step + 1]) / (step + 1)
        if average <= 4.0:
            assert not trial.should_prune()


@pytest.mark.parametrize("step_offset", [0, 2**53, 2**60])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_wilcoxon_pruner_preserves_integer_step_ids(
    step_offset: int, direction: Literal["minimize", "maximize"]
) -> None:
    p_threshold = 0.01
    study = optuna.create_study(
        direction=direction,
        pruner=optuna.pruners.WilcoxonPruner(p_threshold=p_threshold),
    )
    study.add_trial(
        optuna.trial.create_trial(
            value=0.0,
            intermediate_values={step_offset + step: 0.0 for step in range(8)},
        )
    )
    trial = study.ask()
    value = 1.0 if direction == "minimize" else -1.0
    for n_reported, step in enumerate(reversed(range(8)), start=1):
        trial.report(value, step_offset + step)
        # All paired differences have the worse sign, whose exact one-sided probability is 2^-n.
        assert trial.should_prune() == (2.0**-n_reported < p_threshold)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_wilcoxon_pruner_pairs_and_orders_integer_step_ids(
    direction: Literal["minimize", "maximize"],
) -> None:
    differences = [1, -2, 3, 4, 5, 6, 7, 8]
    positive_rank_sum = sum(rank for rank in differences if rank > 0)
    exact_p = sum(
        sum(rank for rank, positive in enumerate(signs, start=1) if positive) >= positive_rank_sum
        for signs in itertools.product([False, True], repeat=len(differences))
    ) / 2 ** len(differences)
    step_offset = 2**60
    sign = 1 if direction == "minimize" else -1
    study = optuna.create_study(
        direction=direction,
        pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.05),
    )
    study.add_trial(
        optuna.trial.create_trial(
            value=35.0,
            intermediate_values={step_offset + step: 10.0 * step for step in reversed(range(8))},
        )
    )
    trial = study.ask()
    for step in [7, 1, 6, 0, 5, 2, 4, 3]:
        trial.report(10.0 * step + sign * differences[step], step_offset + step)

    with patch("optuna.pruners._wilcoxon.ss.wilcoxon", wraps=stats.wilcoxon) as wilcoxon:
        assert trial.should_prune() == (exact_p < 0.05)
    assert wilcoxon.call_args is not None
    np.testing.assert_array_equal(wilcoxon.call_args.args[0], np.array(differences) * sign)
