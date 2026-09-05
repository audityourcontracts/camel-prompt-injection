# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comprehension filters retain assignment-expression effects on every exit."""

import ast

import pytest

from camel.capabilities import Capabilities, get_all_readers, get_all_sources, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.security_policy import NoSecurityPolicyEngine


def _eval(
    code: str,
    namespace: ns.Namespace | None = None,
    mode: interpreter.MetadataEvalMode = interpreter.MetadataEvalMode.NORMAL,
):
    return interpreter.camel_eval(
        ast.parse(code),
        namespace if namespace is not None else ns.Namespace.with_builtins(),
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), mode),
    )


def _raw(evaluated, name: str):
    item = evaluated.namespace.get(name)
    assert item is not None
    return item.raw


@pytest.mark.parametrize(
    ("expression", "empty_result"),
    (
        ("[x for x in [1] if (y := '')]", []),
        ("{x for x in [1] if (y := '')}", set()),
        ("{x: x for x in [1] if (y := '')}", {}),
    ),
)
def test_false_filter_assignment_persists_for_every_comprehension(expression, empty_result):
    evaluated = _eval(f"y = 'old'\nout = {expression}")
    assert isinstance(evaluated.result, result.Ok)
    assert _raw(evaluated, "y") == ""
    assert _raw(evaluated, "out") == empty_result


def test_assignments_in_earlier_and_rejecting_filters_both_persist():
    evaluated = _eval("y = 'old'\nz = 'old-z'\nout = [x for x in [1] if (y := 'yes') if (z := '')]")
    assert isinstance(evaluated.result, result.Ok)
    assert _raw(evaluated, "y") == "yes"
    assert _raw(evaluated, "z") == ""
    assert _raw(evaluated, "out") == []


def test_last_rejected_iteration_retains_its_assignment():
    evaluated = _eval("y = 'old'\nout = [x for x in [1, 2] if (y := ('yes' if x == 1 else ''))]")
    assert isinstance(evaluated.result, result.Ok)
    assert _raw(evaluated, "y") == ""
    assert _raw(evaluated, "out") == [1]


def test_false_filter_assignment_persists_in_nested_generator():
    evaluated = _eval("y = 'old'\nout = [(x, z) for x in [1] for z in [2] if (y := '')]")
    assert isinstance(evaluated.result, result.Ok)
    assert _raw(evaluated, "y") == ""
    assert _raw(evaluated, "out") == []


def test_earlier_filter_assignment_persists_when_later_filter_errors():
    evaluated = _eval("y = 'old'\nout = [x for x in [1] if (y := 'yes') if missing]")
    assert isinstance(evaluated.result, result.Error)
    assert _raw(evaluated, "y") == "yes"


def test_assignment_prefix_in_failing_filter_persists():
    evaluated = _eval("y = 'old'\nout = [x for x in [1] if (y := 'yes') and missing]")
    assert isinstance(evaluated.result, result.Error)
    assert _raw(evaluated, "y") == "yes"


@pytest.mark.parametrize("mode", (interpreter.MetadataEvalMode.NORMAL, interpreter.MetadataEvalMode.STRICT))
@pytest.mark.parametrize(
    "expression",
    ("[x for x in [1] if CONDITION]", "{x for x in [1] if CONDITION}", "{x: x for x in [1] if CONDITION}"),
)
def test_iteration_target_is_still_restored_on_false_and_error_exits(expression, mode):
    metadata = Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"reviewer"}))
    dependency = value.CaMeLStr.from_raw(
        "dependency",
        Capabilities(frozenset({sources.SourceEnum.Assistant}), frozenset({"reviewer", "user"})),
        (),
    )
    outer = value.CaMeLStr.from_raw("outer", metadata, (dependency,))
    namespace = ns.Namespace.with_builtins({"x": outer})
    original_names = set(namespace.variables)

    for condition, expected_result in (("False", result.Ok), ("missing", result.Error)):
        evaluated = _eval("out = " + expression.replace("CONDITION", condition), namespace, mode)
        assert isinstance(evaluated.result, expected_result)
        restored = evaluated.namespace.get("x")
        assert restored is not None
        assert restored is outer
        assert restored.raw == "outer"
        assert restored.metadata == metadata
        assert restored._dependencies == (dependency,)
        assert get_all_sources(restored)[0] == frozenset({sources.SourceEnum.User, sources.SourceEnum.Assistant})
        assert get_all_readers(restored)[0] == frozenset({"reviewer"})
        assert namespace.get("x") is outer
        assert set(namespace.variables) == original_names


@pytest.mark.parametrize("mode", (interpreter.MetadataEvalMode.NORMAL, interpreter.MetadataEvalMode.STRICT))
@pytest.mark.parametrize(
    "expression",
    ("[x for x in [1] if CONDITION]", "{x for x in [1] if CONDITION}", "{x: x for x in [1] if CONDITION}"),
)
def test_unbound_iteration_target_is_removed_on_false_and_error_exits(expression, mode):
    namespace = ns.Namespace.with_builtins()
    original_names = set(namespace.variables)
    for condition, expected_result in (("False", result.Ok), ("missing", result.Error)):
        evaluated = _eval("out = " + expression.replace("CONDITION", condition), namespace, mode)
        assert isinstance(evaluated.result, expected_result)
        assert evaluated.namespace.get("x") is None
        assert namespace.get("x") is None
        assert set(namespace.variables) == original_names


@pytest.mark.parametrize("mode", (interpreter.MetadataEvalMode.NORMAL, interpreter.MetadataEvalMode.STRICT))
@pytest.mark.parametrize("filter_suffix", ("", " or missing"))
def test_filter_assignment_preserves_assigned_metadata_on_false_and_error_exits(mode, filter_suffix):
    metadata = Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"reviewer"}))
    assigned = value.CaMeLStr.from_raw("", metadata, ())
    previous = value.CaMeLStr.from_raw(
        "old", Capabilities(frozenset({sources.SourceEnum.Assistant}), frozenset({"other"})), ()
    )
    namespace = ns.Namespace.with_builtins({"incoming": assigned, "y": previous})

    evaluated = _eval(f"out = [x for x in [1] if (y := incoming){filter_suffix}]", namespace, mode)

    assert isinstance(evaluated.result, result.Error if filter_suffix else result.Ok)
    retained = evaluated.namespace.get("y")
    assert retained is not None
    assert retained.raw == ""
    assert retained.metadata == metadata
    assert get_all_sources(retained)[0] == frozenset({sources.SourceEnum.User})
    assert get_all_readers(retained)[0] == frozenset({"reviewer"})
    assert evaluated.namespace.get("x") is None
    assert namespace.get("y") is previous
    assert assigned.metadata == metadata
