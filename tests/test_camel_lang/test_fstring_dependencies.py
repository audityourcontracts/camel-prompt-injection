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

"""F-string results depend on every evaluated component, including components
whose string representation is empty.
"""

import ast

from camel.capabilities import Capabilities, get_all_readers, get_all_sources, readers, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter.namespace import Namespace
from camel.security_policy import NoSecurityPolicyEngine

_PRIVATE = Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"user"}))


def _namespace_with_private(empty: bool) -> Namespace:
    namespace = Namespace()
    namespace.set_variable(
        "private",
        value.CaMeLStr.from_raw("" if empty else "confidential", _PRIVATE, ()),
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


def _get(namespace: Namespace, var: str) -> value.CaMeLValue:
    got = namespace.get(var)
    assert got is not None, f"{var} not bound"
    return got


def test_nonempty_formatted_value_keeps_readers():
    ev = _eval('res = f"see: {private}"', _namespace_with_private(empty=False))
    assert isinstance(ev.result, result.Ok)
    formatted = _get(ev.namespace, "res")
    assert formatted.raw == "see: confidential"
    effective_readers, _ = get_all_readers(formatted)
    assert effective_readers == frozenset({"user"})


def test_empty_rendering_formatted_value_keeps_readers():
    ev = _eval('res = f"a{private}b"', _namespace_with_private(empty=True))
    assert isinstance(ev.result, result.Ok)
    assert _get(ev.namespace, "res").raw == "ab"
    effective_readers, _ = get_all_readers(_get(ev.namespace, "res"))
    assert effective_readers == frozenset({"user"})


def test_entirely_empty_fstring_keeps_component_metadata():
    ev = _eval('res = f"{private}"', _namespace_with_private(empty=True))
    assert isinstance(ev.result, result.Ok)
    formatted = _get(ev.namespace, "res")
    assert formatted.raw == ""
    actual_sources, _ = get_all_sources(formatted)
    actual_readers, _ = get_all_readers(formatted)
    assert actual_sources == frozenset({sources.SourceEnum.CaMeL, sources.SourceEnum.User})
    assert actual_readers == frozenset({"user"})


def test_multiple_fstring_components_combine_labels_including_empty_component():
    left_source = sources.Tool(
        "left-source",
        frozenset({sources.SourceEnum.TrustedToolSource}),
    )
    namespace = Namespace(
        {
            "left": value.CaMeLStr.from_raw(
                "alpha",
                Capabilities(frozenset({left_source}), frozenset({"reviewer", "user"})),
                (),
            ),
            "empty": value.CaMeLStr.from_raw(
                "",
                Capabilities(frozenset({sources.SourceEnum.Assistant}), readers.Public()),
                (),
            ),
            "right": value.CaMeLStr.from_raw(
                "omega",
                Capabilities(frozenset({sources.SourceEnum.User}), frozenset({"user"})),
                (),
            ),
        }
    )

    ev = _eval('res = f"{left}|{empty}|{right}"', namespace)
    assert isinstance(ev.result, result.Ok)
    formatted = _get(ev.namespace, "res")
    assert formatted.raw == "alpha||omega"

    actual_sources, _ = get_all_sources(formatted)
    actual_readers, _ = get_all_readers(formatted)
    assert actual_sources == frozenset(
        {
            sources.SourceEnum.CaMeL,
            sources.SourceEnum.User,
            sources.SourceEnum.Assistant,
            left_source,
        }
    )
    assert actual_readers == frozenset({"user"})
