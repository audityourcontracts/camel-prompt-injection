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

"""Class instances retain constructor-argument labels through construction
and raw-field reads.
"""

import ast

from agentdojo import functions_runtime
from pydantic import BaseModel

from camel.capabilities import Capabilities, get_all_readers, get_all_sources, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.pipeline_elements.agentdojo_function import make_agentdojo_namespace
from camel.security_policy import NoSecurityPolicyEngine

_PRIVATE = Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"user"}))
_BALANCE_LIKE = Capabilities(frozenset({sources.Tool("get_balance", frozenset())}), frozenset())


class Receipt(BaseModel):
    amount: float


def get_receipt() -> Receipt:
    """Get the latest receipt."""
    return Receipt(amount=100.0)


def _namespace() -> ns.Namespace:
    runtime = functions_runtime.FunctionsRuntime()
    runtime.register_function(get_receipt)
    namespace = ns.Namespace.with_builtins()
    namespace = namespace.add_variables(make_agentdojo_namespace(namespace, runtime, None))
    namespace.set_variable("private_amount", value.CaMeLFloat(1000.0, _PRIVATE, ()))
    namespace.set_variable("balance", value.CaMeLFloat(1000.0, _BALANCE_LIKE, ()))
    return namespace


def _eval(code: str, namespace: ns.Namespace):
    return interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )


def _get(namespace: ns.Namespace, var: str) -> value.CaMeLValue:
    got = namespace.get(var)
    assert got is not None, f"{var} not bound"
    return got


def _build_doc(namespace: ns.Namespace, source_var: str = "private_amount") -> ns.Namespace:
    for stmt in ("class Doc(BaseModel):\n    amount: float", f"doc = Doc(amount={source_var})"):
        ev = _eval(stmt, namespace)
        assert isinstance(ev.result, result.Ok), f"setup failed: {stmt}"
        namespace = ev.namespace
    return namespace


def test_instance_aggregates_ctor_arg_readers():
    namespace = _build_doc(_namespace())
    readers, _ = get_all_readers(_get(namespace, "doc"))
    assert readers == frozenset({"user"})


def test_raw_field_read_keeps_instance_readers():
    namespace = _build_doc(_namespace())
    ev = _eval("a = doc.amount", namespace)
    assert isinstance(ev.result, result.Ok)
    readers, _ = get_all_readers(_get(ev.namespace, "a"))
    assert readers == frozenset({"user"})


def test_wrapped_ctor_arg_is_in_instance_dependencies():
    namespace = _build_doc(_namespace())
    doc = _get(namespace, "doc")
    ctor_arg = _get(namespace, "private_amount")
    dependencies, _ = doc.get_dependencies()
    assert ctor_arg in dependencies, "the wrapped constructor argument must be an instance dependency"


def test_raw_field_read_depends_directly_on_instance():
    namespace = _build_doc(_namespace())
    ev = _eval("a = doc.amount", namespace)
    assert isinstance(ev.result, result.Ok)
    doc = _get(ev.namespace, "doc")
    a = _get(ev.namespace, "a")
    dependencies, _ = a.get_dependencies()
    assert doc in dependencies, "the raw-field fallback must carry the instance as a dependency"


def test_ctor_arg_provenance_reaches_raw_field():
    namespace = _build_doc(_namespace(), source_var="balance")
    ev = _eval("a = doc.amount", namespace)
    assert isinstance(ev.result, result.Ok)
    srcs, _ = get_all_sources(_get(ev.namespace, "a"))
    assert any(isinstance(s, sources.Tool) and s.tool_name == "get_balance" for s in srcs), (
        "the constructor argument's provenance must survive construction and field read"
    )


def test_tool_returned_instance_field_keeps_tool_source():
    namespace = _namespace()
    ev = _eval("amt = get_receipt().amount", namespace)
    assert isinstance(ev.result, result.Ok)
    srcs, _ = get_all_sources(_get(ev.namespace, "amt"))
    assert any(isinstance(s, sources.Tool) and s.tool_name == "get_receipt" for s in srcs), (
        "control: tool-instance field must keep the tool source"
    )
