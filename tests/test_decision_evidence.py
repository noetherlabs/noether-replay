"""Independent exact oracles for the small public identification example."""

from __future__ import annotations

import copy
import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from noether_replay.model import Recording, canonical_json
from noether_replay.timing import qualify_command_timing

_SPEC = importlib.util.spec_from_file_location(
    "decision_evidence_example", Path(__file__).parents[1] / "examples" / "decision_evidence.py",
)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def number(value):
    """Independent test decoder, not the method's rational reader."""
    return Fraction(value["numerator"], value["denominator"])


def visible_bytes(case):
    recording = case["recording"]
    return canonical_json({
        "manifest": recording.manifest(),
        "array_bytes": {stream.id: stream.values.tobytes().hex() for stream in recording.streams},
    })


@pytest.mark.parametrize("y,a,g1,g2", [
    (12, 3, Fraction(1), Fraction(1, 2)),
    (-7, 2, Fraction(3, 7), Fraction(5, 3)),
    (Fraction(2, 9), Fraction(-4, 3), Fraction(11, 2), Fraction(1, 17)),
])
def test_gain_alias_is_exact_with_distinct_residuals_and_identical_visible_bytes(y, a, g1, g2):
    first, second = (example.gain_case(y, a, gain) for gain in (g1, g2))
    assert visible_bytes(first) == visible_bytes(second)
    for case, gain in ((first, g1), (second, g2)):
        hidden_x = number(case["truth"]["disturbance"])
        # Derive the observation independently from each hidden physical world.
        assert gain * (hidden_x - a) == y
        assert number(case["truth"]["residual"]) == hidden_x - a
    assert number(first["truth"]["residual"]) != number(second["truth"]["residual"])


@pytest.mark.parametrize("alternative_gain", [Fraction(1, 100), Fraction(7, 3), Fraction(999)])
def test_gain_compatible_set_is_not_restricted_to_displayed_worlds(alternative_gain):
    original = example.gain_case(12, 3, 1)
    other = example.gain_case(12, 3, alternative_gain)
    assert visible_bytes(original) == visible_bytes(other)
    output = example.constraint_baseline(original["recording"])
    assert output["supported_set"] == {"kind": "positive_reals"}
    assert output["status"] == "set_valued"
    assert number(other["truth"]["target_value"]) == alternative_gain


@pytest.mark.parametrize("y,g", [(12, Fraction(1, 2)), (-7, Fraction(3, 5))])
def test_independent_nonzero_residual_resolves_gain(y, g):
    case = example.gain_case(y, 3, g, reference=True)
    output = example.constraint_baseline(case["recording"])
    assert number(output["supported_set"]["value"]) == g
    assert output["status"] == "resolved"
    result = example.evaluate_case(case, example.constraint_baseline)
    assert result["scores"]["compatible_set_covered"] is True
    assert result["scores"]["incorrect_unique"] is False
    assert result["scores"]["unsupported_unique"] is False


def test_zero_reference_does_not_identify_positive_gain():
    first, second = (example.gain_case(0, 3, g, reference=True) for g in (1, Fraction(1, 2)))
    assert visible_bytes(first) == visible_bytes(second)
    output = example.constraint_baseline(first["recording"])
    assert output["supported_set"] == {"kind": "positive_reals"}
    assert output["status"] == "set_valued"
    assert output["reason"] == "zero_residual_is_uninformative"


@pytest.mark.parametrize("y,q,a1,a2", [(7, 5, 5, 0), (-4, 0, 8, -3),
                                     (Fraction(2, 3), Fraction(7, 5), Fraction(-3, 8), 90)])
def test_actuation_alias_is_exact_with_identical_requests_and_wfs(y, q, a1, a2):
    first, second = (example.actuation_case(y, q, a) for a in (a1, a2))
    assert visible_bytes(first) == visible_bytes(second)
    for case, applied in ((first, a1), (second, a2)):
        assert number(case["truth"]["disturbance"]) - applied == y
    assert number(first["truth"]["target_value"]) != number(second["truth"]["target_value"])
    assert example.constraint_baseline(first["recording"])["supported_set"] == {"kind": "reals"}


@pytest.mark.parametrize("alternative_applied", [Fraction(-100), Fraction(9, 13), Fraction(999)])
def test_unobserved_actuation_allows_values_beyond_zero_and_the_request(alternative_applied):
    first = example.actuation_case(7, 5, 5)
    other = example.actuation_case(7, 5, alternative_applied)
    assert visible_bytes(first) == visible_bytes(other)
    assert other["truth"]["compatible_set"] == {"kind": "reals"}


@pytest.mark.parametrize("applied,match", [(5, "matches"), (0, "differs"), (Fraction(5, 2), "differs")])
def test_independent_readback_resolves_applied_value_and_value_equality(applied, match):
    case = example.actuation_case(7, 5, applied, readback=True)
    result = example.evaluate_case(case, example.constraint_baseline)
    assert number(result["decision"]["supported_set"]["value"]) == applied
    assert result["decision"]["requested_value_match"] == match
    assert result["scores"]["requested_value_equality_correct"] is True
    assert result["scores"]["requested_value_equality_supported"] is True
    assert result["scores"]["transaction_causation_assessed"] is False


def test_zero_request_readback_identifies_value_not_transaction_execution():
    executed = example.actuation_case(7, 0, 0, readback=True, executed=True)
    unchanged = example.actuation_case(7, 0, 0, readback=True, executed=False)
    assert visible_bytes(executed) == visible_bytes(unchanged)
    assert executed["truth"]["transaction_executed"] is True
    assert unchanged["truth"]["transaction_executed"] is False
    first = example.evaluate_case(executed, example.constraint_baseline)
    second = example.evaluate_case(unchanged, example.constraint_baseline)
    assert first == second
    assert first["decision"]["requested_value_match"] == "matches"
    assert first["scores"]["requested_value_equality_supported"] is True
    assert first["scores"]["transaction_causation_assessed"] is False


def test_readback_value_does_not_create_effect_timing():
    record = example.actuation_case(7, 5, 5, readback=True)["recording"]
    timing = qualify_command_timing(
        record, applied_stream_id="readback",
        associations=[{"applied_index": 0, "observations": [{"stream_id": "wfs", "sample_index": 0}]}],
    )
    assert timing["commands"][0]["causal_order"] == "not_established"
    assert "missing_effective_time" in timing["commands"][0]["reasons"]
    with pytest.raises(ValueError, match="command_applied"):
        qualify_command_timing(record, applied_stream_id="requested", associations=[])


def test_matching_realized_world_is_not_unique_identification():
    for case in (example.gain_case(12, 3, 1), example.actuation_case(7, 5, 5)):
        result = example.evaluate_case(case, example.nominal_guess)
        assert result["scores"]["realized_target_in_set"] is True
        assert result["scores"]["incorrect_unique"] is False
        assert result["scores"]["compatible_set_covered"] is False
        assert result["scores"]["unsupported_unique"] is True


def test_wrong_guess_and_unsupported_guess_are_distinct_metrics():
    result = example.evaluate_case(example.gain_case(12, 3, Fraction(1, 2)), example.nominal_guess)
    assert result["scores"]["incorrect_unique"] is True
    assert result["scores"]["unsupported_unique"] is True
    result = example.evaluate_case(example.gain_case(12, 3, Fraction(1, 2), reference=True),
                                   example.nominal_guess)
    assert result["scores"]["incorrect_unique"] is True
    assert result["scores"]["compatible_set_covered"] is False


def test_value_equality_is_scored_separately_from_applied_value_estimation():
    case = example.actuation_case(7, 5, 3, readback=True)

    def wrong_value_but_correct_difference(record):
        return example.decision(record, example.singleton(4), "deliberate test claim")

    result = example.evaluate_case(case, wrong_value_but_correct_difference)
    assert result["scores"]["incorrect_unique"] is True
    assert result["scores"]["requested_value_equality_correct"] is True
    assert result["scores"]["requested_value_equality_supported"] is True
    assert result["scores"]["transaction_causation_assessed"] is False


def test_hidden_truth_and_case_identifiers_are_not_passed_to_callback():
    seen = []

    def capture(record):
        assert isinstance(record, Recording)
        seen.append(canonical_json(record.manifest()))
        return example.always_withhold(record)

    cases = [example.gain_case(12, 3, gain) for gain in (1, Fraction(1, 2))]
    for case in cases:
        case["case_id"] = "scoring-only-private-label"
        case["group_id"] = "scoring-only-private-group"
        before = visible_bytes(case)
        assert example.evaluate_case(case, capture)["terminal_status"] == "completed"
        assert visible_bytes(case) == before
    assert seen[0] == seen[1]
    assert b"scoring-only" not in seen[0]
    assert b"target_value" not in seen[0]
    assert b"compatible_set" not in seen[0]


def test_source_arrays_are_detached_and_callback_mutation_fails():
    case = example.gain_case(12, 3, 1)
    before = visible_bytes(case)

    def mutate(record):
        output = example.always_withhold(record)
        record.streams[0].values[0, 0] += 1
        return output

    result = example.evaluate_case(case, mutate)
    assert result["terminal_status"] == "failed"
    assert visible_bytes(case) == before


@pytest.mark.parametrize("malformation", ["missing", "status", "execution_claim", "gain_domain", "nonfinite"])
def test_malformed_decisions_remain_terminal_failures(malformation):
    case = example.gain_case(12, 3, 1)

    def malformed(record):
        output = example.nominal_guess(record)
        if malformation == "missing":
            return None
        if malformation == "status":
            output["status"] = "withheld"
        elif malformation == "execution_claim":
            output["transaction_causation_assessed"] = True
        elif malformation == "gain_domain":
            output["supported_set"] = example.singleton(-1)
        else:
            output["supported_set"]["value"] = float("nan")
        return output

    result = example.evaluate_case(case, malformed)
    assert result["terminal_status"] == "failed"
    assert result["decision"] is None
    assert result["scores"] is None


def test_callback_exception_preserves_coverage_and_does_not_publish_its_message():
    def fail(_record):
        raise RuntimeError("private exception text")

    output = example.report(methods={"failure": fail})
    assert all(row["terminal_status"] == "failed" for row in output["cases"])
    for summary in output["summary"]:
        assert summary["terminal_failures"] == summary["case_count"]
        assert summary["resolution_coverage"] == {"numerator": 0, "denominator": summary["case_count"]}
        assert summary["selective_realized_error"] is None
    assert b"private exception text" not in canonical_json(output)


def test_always_withhold_has_no_resolution_or_falsely_zero_selective_error():
    output = example.report(methods={"withhold": example.always_withhold})
    for summary in output["summary"]:
        assert summary["resolved_count"] == 0
        assert summary["selective_realized_error"] is None
        assert summary["returned_set_count"] == 0
        assert summary["identifiable_case_count"] > 0
        assert summary["correct_resolution_on_identifiable"]["numerator"] == 0


def test_full_compatible_sets_get_coverage_but_not_resolution_credit():
    case = example.gain_case(12, 3, 1)
    result = example.evaluate_case(case, example.constraint_baseline)
    assert result["scores"]["compatible_set_covered"] is True
    assert result["scores"]["realized_target_in_set"] is True
    assert result["scores"]["resolved"] is False


@pytest.mark.parametrize("mutation", ["array", "units", "model", "kind"])
def test_public_scalar_contract_disagreement_is_not_silently_reinterpreted(mutation):
    record = example.gain_case(12, 3, 1)["recording"]
    if mutation == "array":
        record.streams[0].values[0, 0] += 1
    elif mutation == "units":
        record.streams[0].units = "radian"
    elif mutation == "model":
        record.metadata["disturbance"] = "known"
    else:
        record.streams[1].kind = "command_requested"
    with pytest.raises(ValueError):
        example.constraint_baseline(record)


@pytest.mark.parametrize("value", [True, 1.0, "1", float("nan")])
def test_exact_fixture_constructor_rejects_implicit_inexact_values(value):
    with pytest.raises(ValueError):
        example.rational(value)


@pytest.mark.parametrize("payload", [
    {"numerator": 1, "denominator": 0}, {"numerator": 2, "denominator": 2},
    {"numerator": True, "denominator": 1}, {"numerator": 1, "denominator": -1},
])
def test_noncanonical_exact_number_rejected(payload):
    with pytest.raises(ValueError):
        example.fraction(payload)


def test_independent_gain_and_actuation_enrichments_change_only_the_new_stream():
    for builder, args, keyword in ((example.gain_case, (12, 3, Fraction(1, 2)), "reference"),
                                   (example.actuation_case, (7, 5, 0), "readback")):
        base = builder(*args)["recording"]
        enriched = builder(*args, **{keyword: True})["recording"]
        assert base.id == enriched.id
        assert base.metadata == enriched.metadata
        assert len(enriched.streams) == len(base.streams) + 1
        for left, right in zip(base.streams, enriched.streams):
            assert canonical_json(left.manifest()) == canonical_json(right.manifest())
            assert np.array_equal(left.values, right.values)


def test_all_aliases_and_controls_share_their_base_construction_group():
    cases = example.cases()
    groups = {}
    for case in cases:
        groups.setdefault(case["group_id"], []).append(case)
    assert len(groups["gain_nonzero"]) == 4
    assert len(groups["actuation_nonzero"]) == 4
    assert len(groups["gain_zero"]) == 2
    assert len(groups["actuation_zero"]) == 2
    for group in groups.values():
        assert len({case["recording"].metadata["target"] for case in group}) == 1


def test_report_deterministic_and_does_not_mutate_cases():
    cases = example.cases()
    before = [visible_bytes(case) for case in cases]
    output = example.report(cases)
    assert canonical_json(output) == canonical_json(example.report(copy.deepcopy(cases)))
    assert [visible_bytes(case) for case in cases] == before
    assert len(output["cases"]) == len(cases) * 3
    assert all(summary["case_count"] == 6 for summary in output["summary"])
    assert all(not row["scores"]["transaction_causation_assessed"] for row in output["cases"])
    assert "confidence_interval" not in canonical_json(output).decode()


def test_json_bound_rejects_report_without_truncation(monkeypatch):
    monkeypatch.setattr(example, "MAX_JSON_BYTES", 100)
    with pytest.raises(ValueError, match="report exceeds"):
        example.report()
