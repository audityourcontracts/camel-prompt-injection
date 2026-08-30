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

"""Container dependency walks include their elements' labels.

`CaMeLIterable.get_dependencies` includes each element and its transitive
dependencies. `CaMeLMapping.get_dependencies` does the same for keys and values.
"""

import ast

from camel.capabilities import Capabilities, get_all_readers, get_all_sources, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter.namespace import Namespace
from camel.security_policy import NoSecurityPolicyEngine

_PRIVATE = Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"user"}))


def _namespace_with_private() -> Namespace:
    namespace = Namespace()
    namespace.set_variable("private", value.CaMeLStr.from_raw("confidential", _PRIVATE, ()))
    return namespace


def _eval(code: str, namespace: Namespace):
    return interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )


def _readers(ev, var: str):
    assert isinstance(ev.result, result.Ok)
    got = ev.namespace.get(var)
    assert got is not None, f"{var} not bound"
    readers, _ = get_all_readers(got)
    return readers


def test_bare_value_keeps_its_readers():
    got = _namespace_with_private().get("private")
    assert got is not None
    readers, _ = get_all_readers(got)
    assert readers == frozenset({"user"})


def test_list_literal_keeps_element_readers():
    ev = _eval("res = [private]", _namespace_with_private())
    assert _readers(ev, "res") == frozenset({"user"})


def test_tuple_literal_keeps_element_readers():
    ev = _eval("res = (private,)", _namespace_with_private())
    assert _readers(ev, "res") == frozenset({"user"})


def test_set_literal_keeps_element_readers():
    ev = _eval("res = {private}", _namespace_with_private())
    assert _readers(ev, "res") == frozenset({"user"})


def test_dict_literal_keeps_value_readers():
    ev = _eval('res = {"k": private}', _namespace_with_private())
    assert _readers(ev, "res") == frozenset({"user"})


def test_dict_literal_keeps_key_readers():
    ev = _eval('res = {private: "v"}', _namespace_with_private())
    assert _readers(ev, "res") == frozenset({"user"})


def test_nested_container_keeps_child_and_explicit_dependency_labels():
    dependency = value.CaMeLStr.from_raw(
        "dependency",
        Capabilities(frozenset({sources.SourceEnum.Assistant}), frozenset({"user"})),
        (),
    )
    child = value.CaMeLStr.from_raw("secret", _PRIVATE, (dependency,))
    nested = value.CaMeLTuple((child,), Capabilities.camel(), ())
    key = value.CaMeLStr.from_raw("key", Capabilities.camel(), ())
    container = value.CaMeLDict({key: nested}, Capabilities.camel(), ())
    unrelated = value.CaMeLStr.from_raw(
        "other",
        Capabilities(frozenset({sources.Tool("unrelated")}), frozenset({"other"})),
        (),
    )
    namespace = Namespace({"container": container, "unrelated": unrelated})

    ev = _eval("res = container", namespace)
    assert isinstance(ev.result, result.Ok)
    got = ev.namespace.get("res")
    assert isinstance(got, value.CaMeLDict)
    assert got.raw == {"key": ("secret",)}

    actual_readers, _ = get_all_readers(got)
    actual_sources, _ = get_all_sources(got)
    assert actual_readers == frozenset({"user"})
    assert actual_sources == frozenset(
        {sources.SourceEnum.CaMeL, sources.SourceEnum.User, sources.SourceEnum.Assistant}
    )
    assert not any(isinstance(source, sources.Tool) and source.tool_name == "unrelated" for source in actual_sources)

    dependencies, _ = got.get_dependencies()
    assert any(item is key for item in dependencies)
    assert any(item is nested for item in dependencies)
    assert any(item is child for item in dependencies)
    assert any(item is dependency for item in dependencies)
    assert all(item is not unrelated for item in dependencies)
    assert got.metadata == Capabilities.camel()
