"""Exact scalar identification examples, not a sampled diagnostic evaluation.

Run with ``python examples/decision_evidence.py``. All quantities are synthetic
rational witnesses. No instruments, random draws, model calls or files are used.
"""

from __future__ import annotations

import copy
import json
import math
from fractions import Fraction

import numpy as np

from noether_replay.model import MAX_JSON_BYTES, Recording, Stream, canonical_json

SCHEMA = "noether-replay-decision-evidence-example/1"
GAIN = "sensor_gain"
APPLIED = "applied_command_value"


def rational(value):
    """Canonical exact JSON number for these mathematical fixtures."""
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValueError("Fixture values must be integers or exact Fractions")
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("Expected a canonical rational number")
    n, d = value["numerator"], value["denominator"]
    if type(n) is not int or type(d) is not int or d <= 0:
        raise ValueError("Rational endpoints require integer numerator and positive denominator")
    result = Fraction(n, d)
    if rational(result) != value:
        raise ValueError("Rational numbers must be reduced")
    return result


def singleton(value):
    return {"kind": "singleton", "value": rational(value)}


def stream(name, kind, value, units):
    exact = rational(value)
    rendered = float(fraction(exact))
    if not math.isfinite(rendered):
        raise ValueError("Fixture value has no finite float64 rendering")
    return Stream(
        name, kind, np.array([[rendered]], dtype=np.float64), units,
        metadata={"exact_value": exact},
    )


def gain_case(measurement, applied, gain, *, reference=False):
    """One world with y = g(x-a); visible evidence excludes g and x."""
    y, a, g = (fraction(rational(value)) for value in (measurement, applied, gain))
    if g <= 0:
        raise ValueError("Sensor gain must be positive")
    residual = y / g
    sources = [stream("wfs", "wfs_measurements", y, "nm_nominal_opd"),
               stream("applied", "command_applied", a, "nm_opd")]
    if reference:
        sources.append(stream("reference", "science", residual, "nm_opd"))
    recording = Recording(
        "exact-gain-evidence", tuple(sources),
        metadata={"target": GAIN, "model": "y=g*(x-a)",
                  "disturbance": "unrestricted", "gain_domain": "positive_reals",
                  "measurement_contract": "exact rational synthetic values"},
    )
    # This compatible-set oracle comes from the construction, not the method.
    compatible = singleton(g) if reference and residual else {"kind": "positive_reals"}
    truth = {"target_value": rational(g), "disturbance": rational(a + residual),
             "residual": rational(residual), "compatible_set": compatible}
    return {"recording": recording, "truth": truth}


def actuation_case(measurement, requested, applied, *, readback=False, executed=None):
    """One world with y = x-a; requested value is not an actuation measurement.

    ``executed`` distinguishes a zero-valued transaction from an identical
    unchanged state. It is hidden and never scored as identified by readback.
    """
    y, q, a = (fraction(rational(value)) for value in (measurement, requested, applied))
    if executed is not None and (type(executed) is not bool or q != 0 or a != 0):
        raise ValueError("Execution witnesses are defined only for the zero-command control")
    sources = [stream("wfs", "wfs_measurements", y, "nm_opd"),
               stream("requested", "command_requested", q, "nm_opd")]
    if readback:
        sources.append(stream("readback", "command_applied", a, "nm_opd"))
    recording = Recording(
        "exact-actuation-evidence", tuple(sources),
        metadata={"target": APPLIED, "model": "y=x-a", "disturbance": "unrestricted",
                  "tool_status": "success", "measurement_contract": "exact rational synthetic values"},
    )
    truth = {"target_value": rational(a), "disturbance": rational(y + a),
             "compatible_set": singleton(a) if readback else {"kind": "reals"},
             "requested_value_matches": a == q, "transaction_executed": executed}
    return {"recording": recording, "truth": truth}


def values(recording):
    """Read only the exact public samples and check their float64 renderings."""
    recording.validate()
    result = {}
    for source in recording.streams:
        value = fraction(source.metadata["exact_value"])
        if (source.values.shape != (1, 1) or source.values.dtype != np.dtype("float64")
                or not math.isfinite(float(value)) or source.values[0, 0] != float(value)):
            raise ValueError("Exact sample and float64 rendering disagree")
        result[source.id] = value
    target = recording.metadata.get("target")
    required, optional = ({"wfs", "applied"}, {"reference"}) if target == GAIN else (
        {"wfs", "requested"}, {"readback"}
    )
    if target not in {GAIN, APPLIED} or not required <= result.keys() <= required | optional:
        raise ValueError("Unknown analytical evidence contract")
    expected_metadata = {
        "target": target, "model": "y=g*(x-a)" if target == GAIN else "y=x-a",
        "disturbance": "unrestricted", "measurement_contract": "exact rational synthetic values",
        **({"gain_domain": "positive_reals"} if target == GAIN else {"tool_status": "success"}),
    }
    if recording.metadata != expected_metadata:
        raise ValueError("Analytical model assumptions changed")
    expected_kinds = {"wfs": "wfs_measurements", "applied": "command_applied",
                      "requested": "command_requested", "reference": "science",
                      "readback": "command_applied"}
    if any(source.kind != expected_kinds[source.id] for source in recording.streams):
        raise ValueError("Stream meaning does not match the analytical contract")
    if any(source.units != ("nm_nominal_opd" if target == GAIN and source.id == "wfs" else "nm_opd")
           for source in recording.streams):
        raise ValueError("Stream units do not match the analytical contract")
    return result


def decision(recording, supported_set, reason):
    data = values(recording)
    kind = supported_set["kind"]
    status = "resolved" if kind == "singleton" else (
        "withheld" if kind == "unreported" else "set_valued"
    )
    match = "not_applicable"
    if recording.metadata["target"] == APPLIED:
        match = "not_resolved" if kind != "singleton" else (
            "matches" if fraction(supported_set["value"]) == data["requested"] else "differs"
        )
    return {"status": status, "supported_set": supported_set, "reason": reason,
            "requested_value_match": match, "transaction_causation_assessed": False}


def constraint_baseline(recording):
    """Solve the declared scalar constraints, retaining unrestricted nuisance."""
    data = values(recording)
    if recording.metadata["target"] == GAIN:
        if "reference" not in data:
            return decision(recording, {"kind": "positive_reals"}, "independent_residual_missing")
        residual = data["reference"]
        if residual == 0:
            if data["wfs"] != 0:
                raise ValueError("Exact zero residual conflicts with a nonzero measurement")
            return decision(recording, {"kind": "positive_reals"}, "zero_residual_is_uninformative")
        gain = data["wfs"] / residual
        if gain <= 0:
            raise ValueError("Evidence contradicts the positive-gain model")
        return decision(recording, singleton(gain), "independent_residual_identifies_gain")
    if "readback" not in data:
        return decision(recording, {"kind": "reals"}, "independent_readback_missing")
    return decision(recording, singleton(data["readback"]), "physical_value_readback")


def nominal_guess(recording):
    data = values(recording)
    value = Fraction(1) if recording.metadata["target"] == GAIN else data["requested"]
    return decision(recording, singleton(value), "nominal_assumption_not_identification")


def always_withhold(recording):
    return decision(recording, {"kind": "unreported"}, "always_withhold_control")


def _validate_decision(recording, output):
    fields = {"status", "supported_set", "reason", "requested_value_match",
              "transaction_causation_assessed"}
    if not isinstance(output, dict) or set(output) != fields:
        raise ValueError("Decision fields do not match the example contract")
    claim = output["supported_set"]
    if not isinstance(claim, dict) or "kind" not in claim:
        raise ValueError("A structured supported set is required")
    allowed = {"singleton", "unreported", "positive_reals" if recording.metadata["target"] == GAIN
               else "reals"}
    kind = claim["kind"]
    if kind not in allowed or set(claim) != ({"kind", "value"} if kind == "singleton" else {"kind"}):
        raise ValueError("Supported set does not match the target domain")
    if kind == "singleton":
        value = fraction(claim["value"])
        if recording.metadata["target"] == GAIN and value <= 0:
            raise ValueError("A sensor-gain claim must be positive")
    if not isinstance(output["reason"], str) or not output["reason"]:
        raise ValueError("A decision reason is required")
    if output != decision(recording, claim, output["reason"]):
        raise ValueError("Status, value match or transaction scope contradicts the supported set")
    if len(canonical_json(output)) > MAX_JSON_BYTES:
        raise ValueError("Decision exceeds the JSON bound")


def _contains(claim, value):
    if claim["kind"] == "unreported":
        return None
    if claim["kind"] == "reals":
        return True
    if claim["kind"] == "positive_reals":
        return value > 0
    return fraction(claim["value"]) == value


def _covers(claim, compatible):
    if claim["kind"] == "unreported":
        return None
    if compatible["kind"] == "singleton":
        return _contains(claim, fraction(compatible["value"]))
    return claim["kind"] == compatible["kind"]


def evaluate_case(case, method):
    """A trusted deterministic callback receives only a detached Recording.

    This separation is not a Python sandbox or a sealed evaluation: the example
    and its truth are public. Failures remain outcomes and do not leave coverage.
    """
    visible = copy.deepcopy(case["recording"])
    digest = visible.digest()
    try:
        output = method(visible)
        _validate_decision(visible, output)
        if visible.digest() != digest:
            raise ValueError("Method changed the visible evidence")
    except Exception as error:
        return {"terminal_status": "failed", "failure_type": type(error).__name__,
                "recording_sha256": digest, "decision": None, "scores": None}
    truth = case["truth"]
    claim = output["supported_set"]
    resolved = output["status"] == "resolved"
    realized = _contains(claim, fraction(truth["target_value"]))
    coverage = _covers(claim, truth["compatible_set"])
    equality_resolved = output["requested_value_match"] in {"matches", "differs"}
    equality_correct = None
    equality_supported = None
    if equality_resolved:
        equality_correct = (output["requested_value_match"] == "matches") == truth["requested_value_matches"]
        compatible = truth["compatible_set"]
        equality_supported = compatible["kind"] == "singleton" and (
            (output["requested_value_match"] == "matches")
            == (fraction(compatible["value"]) == values(case["recording"])["requested"])
        )
    return {
        "terminal_status": "completed", "recording_sha256": digest,
        "decision": copy.deepcopy(output),
        "scores": {
            "resolved": resolved,
            "realized_target_in_set": realized,
            "compatible_set_covered": coverage,
            "incorrect_unique": resolved and not realized,
            "unsupported_unique": resolved and not coverage,
            "requested_value_equality_resolved": equality_resolved,
            "requested_value_equality_correct": equality_correct,
            "requested_value_equality_supported": equality_supported,
            "transaction_causation_assessed": False,
        },
    }


def cases():
    """Hand-constructed witnesses, not independent sampled experiments."""
    result = []
    for label, gain in (("unit", Fraction(1)), ("half", Fraction(1, 2))):
        for condition, reference, y in (("trace", False, 12), ("reference", True, 12),
                                        ("zero_reference", True, 0)):
            case = gain_case(y, 3, gain, reference=reference)
            case.update({"case_id": f"gain_{condition}_{label}",
                         "group_id": "gain_zero" if y == 0 else "gain_nonzero"})
            result.append(case)
    for label, applied in (("matched", 5), ("unmatched", 0)):
        for condition, readback in (("trace", False), ("readback", True)):
            case = actuation_case(7, 5, applied, readback=readback)
            case.update({"case_id": f"actuation_{condition}_{label}", "group_id": "actuation_nonzero"})
            result.append(case)
    for executed in (False, True):
        case = actuation_case(7, 0, 0, readback=True, executed=executed)
        case.update({"case_id": f"zero_command_execution_{int(executed)}", "group_id": "actuation_zero"})
        result.append(case)
    return result


def _ratio(numerator, denominator):
    return None if denominator == 0 else {"numerator": numerator, "denominator": denominator}


def report(example_cases=None, methods=None):
    example_cases = cases() if example_cases is None else example_cases
    methods = {"constraint": constraint_baseline, "nominal_guess": nominal_guess,
               "always_withhold": always_withhold} if methods is None else methods
    rows = []
    for case in example_cases:
        for name, method in methods.items():
            rows.append({"case_id": case["case_id"], "group_id": case["group_id"],
                         "target": case["recording"].metadata["target"], "method": name,
                         "truth": copy.deepcopy(case["truth"]), **evaluate_case(case, method)})
    summaries = []
    for name in methods:
        for target in (GAIN, APPLIED):
            selected = [row for row in rows if row["method"] == name and row["target"] == target]
            scores = [row["scores"] for row in selected if row["scores"] is not None]
            resolved = sum(score["resolved"] for score in scores)
            incorrect = sum(score["incorrect_unique"] for score in scores)
            unsupported = sum(score["unsupported_unique"] for score in scores)
            identifiable = [row for row in selected if row["truth"]["compatible_set"]["kind"] == "singleton"]
            identified = sum(row["scores"] is not None and row["scores"]["resolved"]
                             and not row["scores"]["incorrect_unique"] for row in identifiable)
            equality = [score for score in scores if score["requested_value_equality_resolved"]]
            summaries.append({
                "method": name, "target": target, "case_count": len(selected),
                "terminal_failures": sum(row["terminal_status"] == "failed" for row in selected),
                "resolved_count": resolved,
                "resolution_coverage": _ratio(resolved, len(selected)),
                "selective_realized_error": _ratio(incorrect, resolved),
                "unsupported_unique_count": unsupported,
                "unsupported_unique_fraction": _ratio(unsupported, resolved),
                "returned_set_count": sum(score["realized_target_in_set"] is not None for score in scores),
                "compatible_set_covered_count": sum(score["compatible_set_covered"] is True for score in scores),
                "identifiable_case_count": len(identifiable),
                "correct_resolution_on_identifiable": _ratio(identified, len(identifiable)),
                "requested_value_equality_resolved_count": len(equality),
                "requested_value_equality_incorrect_count": sum(not score["requested_value_equality_correct"]
                                                               for score in equality),
                "requested_value_equality_unsupported_count": sum(not score["requested_value_equality_supported"]
                                                                 for score in equality),
            })
    result = {
        "schema": SCHEMA, "evidence_type": "deterministic analytical examples",
        "cases": rows, "summary": summaries,
        "limitations": [
            "These are exact scalar constructions, not a sampled diagnostic performance study.",
            "Positive-gain and real applied-value sets retain unrestricted disturbance uncertainty.",
            "Displayed alias witnesses are not an exhaustive list of compatible physical worlds.",
            "Exact independent measurements are fixture assumptions, not instrument accuracy claims.",
            "Float64 arrays render exact rational samples; methods use the public rational contract.",
            "Readback value equality does not establish whether a command transaction executed.",
            "Callbacks are trusted Python; payload separation is not an adversarial sandbox.",
            "No confidence intervals, model superiority, causal benefit or hardware safety is established.",
        ],
    }
    if len(canonical_json(result)) > MAX_JSON_BYTES:
        raise ValueError("Example report exceeds the JSON bound")
    return result


if __name__ == "__main__":
    print(json.dumps(report(), sort_keys=True, indent=2, allow_nan=False))
