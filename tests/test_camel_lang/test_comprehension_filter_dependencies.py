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

"""Comprehension results carry evaluated filter dependencies in STRICT mode only."""

import ast

from camel.capabilities import Capabilities, get_all_readers, is_public, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter.namespace import Namespace
from camel.security_policy import NoSecurityPolicyEngine

_PRIVATE = Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"user"}))

_COMPREHENSION = "filtered = [x for x in public_items if private_data]"
_SET_COMPREHENSION = "filtered = {x for x in public_items if private_data}"
_DICT_COMPREHENSION = "filtered = {x: x for x in public_items if private_data}"
_FALSE_COMPREHENSION = "filtered = [x for x in public_items if private_empty]"
_FOR_IF = """\
filtered = []
for x in public_items:
    if private_data:
        filtered = filtered + [x]
"""


def _namespace() -> Namespace:
    namespace = Namespace()
    namespace.set_variable(
        "public_items",
        value.CaMeLList(
            [
                value.CaMeLStr.from_raw("alpha", Capabilities.default(), ()),
                value.CaMeLStr.from_raw("beta", Capabilities.default(), ()),
            ],
            Capabilities.default(),
            (),
        ),
    )
    namespace.set_variable("private_data", value.CaMeLStr.from_raw("secret", _PRIVATE, ()))
    # A falsy private value (empty string) for the false-filter case.
    namespace.set_variable("private_empty", value.CaMeLStr.from_raw("", _PRIVATE, ()))
    return namespace


def _eval(code: str, namespace: Namespace, mode):
    return interpreter.camel_eval(
        ast.parse(code), namespace, [], [], interpreter.EvalArgs(NoSecurityPolicyEngine(), mode)
    )


def _get(namespace: Namespace, var: str) -> value.CaMeLValue:
    got = namespace.get(var)
    assert got is not None, f"{var} not bound"
    return got


def _readers(ev, var: str):
    assert isinstance(ev.result, result.Ok)
    readers, _ = get_all_readers(_get(ev.namespace, var))
    return readers


def test_listcomp_filter_taints_result_in_strict():
    ev = _eval(_COMPREHENSION, _namespace(), interpreter.MetadataEvalMode.STRICT)
    assert _readers(ev, "filtered") == frozenset({"user"})


def test_setcomp_filter_taints_result_in_strict():
    ev = _eval(_SET_COMPREHENSION, _namespace(), interpreter.MetadataEvalMode.STRICT)
    assert _readers(ev, "filtered") == frozenset({"user"})


def test_dictcomp_filter_taints_result_in_strict():
    ev = _eval(_DICT_COMPREHENSION, _namespace(), interpreter.MetadataEvalMode.STRICT)
    assert _readers(ev, "filtered") == frozenset({"user"})


def test_false_filter_is_collected_in_strict():
    """The filter is collected whether it passes or not: an all-filtered-out
    result is empty and still carries the filter's labels."""
    ev = _eval(_FALSE_COMPREHENSION, _namespace(), interpreter.MetadataEvalMode.STRICT)
    filtered = _get(ev.namespace, "filtered")
    assert filtered.raw == []
    assert _readers(ev, "filtered") == frozenset({"user"})


def test_listcomp_filter_not_propagated_in_normal():
    ev = _eval(_COMPREHENSION, _namespace(), interpreter.MetadataEvalMode.NORMAL)
    assert isinstance(ev.result, result.Ok)
    assert is_public(_get(ev.namespace, "filtered"))


def test_equivalent_for_if_is_tainted_in_strict():
    ev = _eval(_FOR_IF, _namespace(), interpreter.MetadataEvalMode.STRICT)
    assert _readers(ev, "filtered") == frozenset({"user"})
