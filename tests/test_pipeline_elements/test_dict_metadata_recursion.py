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

"""Dict-typed tool outputs carry intended metadata through nested values.

Metadata assignment covers mapping keys and values, nested list/tuple/set
values, and string characters so
dependency walks observe consistent labels at each reachable value.
"""

import ast

import pytest
from agentdojo import functions_runtime

from camel.capabilities import (
    Capabilities,
    get_all_readers,
    get_all_sources,
    is_public,
    is_trusted,
    readers,
    sources,
)
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.pipeline_elements.agentdojo_function import _get_metadata_for_ad, make_agentdojo_namespace
from camel.security_policy import NoSecurityPolicyEngine


def get_restaurants_address() -> dict:
    """Get the addresses of restaurants in the city."""
    return {"data": {"olive": "123 Main St"}}


def get_rating_reviews_for_restaurants() -> dict:
    """Get rating reviews for restaurants."""
    return {"reviews": {"olive": ["Great place!", "Loved it."]}}


def get_user_info() -> dict:
    """Get private user information."""
    return {"name": "Emma"}


def _namespace() -> ns.Namespace:
    runtime = functions_runtime.FunctionsRuntime()
    runtime.register_function(get_restaurants_address)
    runtime.register_function(get_rating_reviews_for_restaurants)
    runtime.register_function(get_user_info)
    namespace = ns.Namespace.with_builtins()
    return namespace.add_variables(make_agentdojo_namespace(namespace, runtime, None))


def _eval(code: str, namespace: ns.Namespace):
    return interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )


def _leaves(mapping):
    for _, v in mapping._python_value.items():
        if isinstance(v._python_value, dict):
            yield from _leaves(v)
        elif isinstance(v._python_value, list):
            yield from v._python_value
        else:
            yield v


def _tool_metadata(
    tool: str,
    inner_sources: frozenset[sources.SourceEnum],
    readers_set: readers.Readers,
) -> Capabilities:
    return Capabilities(frozenset({sources.Tool(tool, inner_sources)}), readers_set)


def _bound(ev: interpreter.EvalResult, name: str) -> value.CaMeLValue:
    assert isinstance(ev.result, result.Ok)
    item = ev.namespace.get(name)
    assert item is not None, f"{name} not bound"
    return item


def _assert_effective_metadata(
    item: value.CaMeLValue,
    expected_sources: frozenset[sources.Source],
    expected_readers: readers.Readers,
) -> None:
    actual_sources, _ = get_all_sources(item)
    actual_readers, _ = get_all_readers(item)
    assert actual_sources == expected_sources
    assert actual_readers == expected_readers


def test_trusted_travel_dict_contents_are_trusted_and_public():
    tool_source = sources.Tool(
        "get_restaurants_address",
        frozenset({sources.SourceEnum.TrustedToolSource}),
    )
    expected_metadata = Capabilities(frozenset({tool_source}), readers.Public())
    ev = _eval("res = get_restaurants_address()", _namespace())
    assert isinstance(ev.result, result.Ok)
    leaves = list(_leaves(ev.namespace.get("res")))
    assert leaves, "expected at least one leaf value"
    for leaf in leaves:
        assert is_trusted(leaf), "trusted travel tool contents must be trusted"
        assert is_public(leaf), "trusted travel tool contents must be public"
        assert leaf.metadata == expected_metadata
        _assert_effective_metadata(leaf, frozenset({tool_source}), readers.Public())


def test_untrusted_travel_dict_contents_keep_user_inner_and_public_readers():
    tool_source = sources.Tool(
        "get_rating_reviews_for_restaurants",
        frozenset({sources.SourceEnum.User}),
    )
    expected_metadata = Capabilities(frozenset({tool_source}), readers.Public())
    ev = _eval("res = get_rating_reviews_for_restaurants()", _namespace())
    assert isinstance(ev.result, result.Ok)
    leaves = list(_leaves(ev.namespace.get("res")))
    assert leaves, "expected at least one leaf value"
    for leaf in leaves:
        assert is_public(leaf), "reviews contents must be public"
        assert leaf.metadata == expected_metadata
        _assert_effective_metadata(leaf, frozenset({tool_source}), readers.Public())


def test_mapping_key_iteration_exposes_tool_metadata():
    tool = "get_restaurants_address"
    expected_metadata = _tool_metadata(
        tool,
        frozenset({sources.SourceEnum.TrustedToolSource}),
        readers.Public(),
    )

    ev = _eval("out = get_restaurants_address()\nfor item in out:\n    selected = item", _namespace())
    selected = _bound(ev, "selected")

    assert isinstance(selected, value.CaMeLStr)
    assert selected.raw == "data"
    assert selected.metadata == expected_metadata
    _assert_effective_metadata(selected, expected_metadata.sources_set, readers.Public())


def test_mapping_lookup_exposes_value_metadata():
    tool = "get_restaurants_address"
    tool_source = sources.Tool(tool, frozenset({sources.SourceEnum.TrustedToolSource}))
    expected_metadata = Capabilities(frozenset({tool_source}), readers.Public())

    ev = _eval('out = get_restaurants_address()\nselected = out["data"]["olive"]', _namespace())
    selected = _bound(ev, "selected")

    assert isinstance(selected, value.CaMeLStr)
    assert selected.raw == "123 Main St"
    assert selected.metadata == expected_metadata
    _assert_effective_metadata(
        selected,
        frozenset({tool_source, sources.SourceEnum.CaMeL, sources.SourceEnum.User}),
        readers.Public(),
    )


def test_private_user_info_lookup_exposes_private_trusted_metadata():
    tool = "get_user_info"
    tool_source = sources.Tool(tool, frozenset({sources.SourceEnum.User}))
    expected_metadata = Capabilities(frozenset({tool_source}), frozenset())

    ev = _eval('out = get_user_info()\nselected = out["name"]', _namespace())
    selected = _bound(ev, "selected")

    assert isinstance(selected, value.CaMeLStr)
    assert selected.raw == "Emma"
    assert selected.metadata == expected_metadata
    _assert_effective_metadata(
        selected,
        frozenset({tool_source, sources.SourceEnum.CaMeL, sources.SourceEnum.User}),
        frozenset(),
    )


def test_mapping_keys_and_values_receive_the_same_boundary_metadata():
    tool = "get_rating_reviews_for_restaurants"
    namespace = ns.Namespace.with_builtins()
    wrapped = value.value_from_raw(
        {"attacker-controlled-key": "review"},
        Capabilities(frozenset({sources.Tool(tool)}), readers.Public()),
        namespace,
        (),
    )

    converted = _get_metadata_for_ad(wrapped, tool)
    key, child = next(iter(converted._python_value.items()))
    expected_metadata = _tool_metadata(tool, frozenset({sources.SourceEnum.User}), readers.Public())
    assert converted.raw == wrapped.raw
    assert converted.metadata == expected_metadata
    assert key.metadata == expected_metadata
    assert child.metadata == expected_metadata


@pytest.mark.parametrize("container_type", (value.CaMeLList, value.CaMeLTuple, value.CaMeLSet))
def test_nested_container_members_receive_boundary_metadata(container_type):
    tool = "get_rating_reviews_for_restaurants"
    metadata = Capabilities.default()
    leaf = value.CaMeLStr.from_raw("review", metadata, ())
    container = container_type([leaf], metadata, ())
    wrapped = value.CaMeLDict(
        {value.CaMeLStr.from_raw("result", metadata, ()): container},
        metadata,
        (),
    )

    converted = _get_metadata_for_ad(wrapped, tool)
    converted_container = next(iter(converted._python_value.values()))
    converted_leaf = next(iter(converted_container._python_value))
    expected_metadata = _tool_metadata(tool, frozenset({sources.SourceEnum.User}), readers.Public())
    assert converted.raw == wrapped.raw
    assert type(converted_container) is container_type
    assert converted_container.metadata == expected_metadata
    assert converted_leaf.metadata == expected_metadata
    assert wrapped.metadata == metadata
    assert container.metadata == metadata
    assert leaf.metadata == metadata


@pytest.mark.parametrize(
    ("container_type", "expected_raw"),
    (
        (value.CaMeLList, []),
        (value.CaMeLTuple, ()),
        (value.CaMeLSet, set()),
    ),
)
def test_empty_top_level_sequences_keep_type_shape_and_root_metadata(container_type, expected_raw):
    tool = "empty_collection_fixture"
    original_metadata = Capabilities.default()
    wrapped = container_type([], original_metadata, ())

    converted = _get_metadata_for_ad(wrapped, tool)

    assert type(converted) is container_type
    assert converted.raw == expected_raw
    assert converted.metadata == _tool_metadata(tool, frozenset(), readers.Public())
    assert wrapped.metadata == original_metadata


def test_relabel_preserves_dependency_only_metadata_and_shared_contents():
    tool = "get_rating_reviews_for_restaurants"
    original_metadata = Capabilities.default()
    dependency = value.CaMeLStr.from_raw("dependency", original_metadata, ())
    shared = value.CaMeLStr.from_raw("shared", original_metadata, ())
    contents = value.CaMeLList([shared, shared], original_metadata, ())
    wrapped = value.CaMeLDict(
        {value.CaMeLStr.from_raw("result", original_metadata, ()): contents},
        original_metadata,
        (dependency,),
    )
    original_key = next(iter(wrapped._python_value))
    original_raw = wrapped.raw

    converted = _get_metadata_for_ad(wrapped, tool)
    converted_key = next(iter(converted._python_value))
    converted_contents = next(iter(converted._python_value.values()))
    expected_metadata = _tool_metadata(tool, frozenset({sources.SourceEnum.User}), readers.Public())
    assert isinstance(converted, value.CaMeLDict)
    assert converted.raw == original_raw
    assert converted.metadata == expected_metadata
    assert converted_key.metadata == expected_metadata
    assert converted_contents.metadata == expected_metadata
    assert converted._dependencies == (dependency,)
    assert converted._dependencies[0] is dependency
    assert dependency.metadata == original_metadata
    assert converted_contents._python_value[0] is converted_contents._python_value[1]
    assert converted_contents._python_value[0].metadata == expected_metadata
    assert wrapped.raw == original_raw
    assert wrapped.metadata == original_metadata
    assert original_key.metadata == original_metadata
    assert contents.metadata == original_metadata
    assert contents._python_value[0] is shared
    assert contents._python_value[1] is shared
    assert shared.metadata == original_metadata
