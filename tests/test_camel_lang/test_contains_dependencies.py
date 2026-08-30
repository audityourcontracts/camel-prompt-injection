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

"""Membership results retain the container's own labels in both polarities,
including `not in` and string containment.
"""

import ast

from camel.capabilities import Capabilities, get_all_readers, readers, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter.namespace import Namespace
from camel.security_policy import NoSecurityPolicyEngine

_PRIVATE = Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"user"}))


def _namespace() -> Namespace:
    namespace = Namespace()
    # Keep container metadata private and elements public so the assertions
    # isolate the dependency on the container itself.
    namespace.set_variable(
        "private_list",
        value.CaMeLList(
            [
                value.CaMeLStr.from_raw("a", Capabilities.default(), ()),
                value.CaMeLStr.from_raw("b", Capabilities.default(), ()),
            ],
            _PRIVATE,
            (),
        ),
    )
    namespace.set_variable(
        "private_str",
        value.CaMeLStr.from_raw("abc", Capabilities.default(), ()).new_with_metadata(_PRIVATE),
    )
    namespace.set_variable("private_empty_list", value.CaMeLList([], _PRIVATE, ()))
    namespace.set_variable(
        "private_empty_str",
        value.CaMeLStr.from_raw("", Capabilities.default(), ()).new_with_metadata(_PRIVATE),
    )
    return namespace


def _eval(code: str, namespace: Namespace):
    return interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )


def _membership_result(code: str) -> tuple[bool, readers.Readers]:
    ev = _eval(code, _namespace())
    assert isinstance(ev.result, result.Ok)
    got = ev.namespace.get("res")
    assert got is not None, "res not bound"
    assert isinstance(got.raw, bool)
    effective_readers, _ = get_all_readers(got)
    return got.raw, effective_readers


def test_true_contains_keeps_container_readers():
    assert _membership_result('res = "a" in private_list') == (True, frozenset({"user"}))


def test_false_contains_keeps_container_readers():
    assert _membership_result('res = "zzz" in private_list') == (False, frozenset({"user"}))


def test_negated_contains_keeps_container_readers():
    assert _membership_result('res = "zzz" not in private_list') == (True, frozenset({"user"}))


def test_false_contains_on_string_keeps_container_readers():
    assert _membership_result('res = "x" in private_str') == (False, frozenset({"user"}))


def test_empty_container_membership_keeps_container_readers():
    assert _membership_result('res = "a" in private_empty_list') == (False, frozenset({"user"}))


def test_empty_string_membership_keeps_container_readers():
    assert _membership_result('res = "x" in private_empty_str') == (False, frozenset({"user"}))
