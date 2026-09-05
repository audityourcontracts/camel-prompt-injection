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

"""Tool outputs carry intended metadata through supported content values.

Metadata assignment covers mapping keys and values, supported sequence
contents, and string characters so extraction and dependency walks observe
the metadata selected for the tool boundary.
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


def get_current_day() -> str:
    """Get the current day."""
    return "Monday"


def get_iban() -> str:
    """Get an empty IBAN fixture."""
    return ""


def reserve_restaurant() -> str:
    """Reserve a restaurant."""
    return "Olive at 19:00"


def read_file() -> str:
    """Read a private file."""
    return "secret"


def get_user_info() -> dict:
    """Get private user information."""
    return {"name": "Emma"}


def get_hotels_address() -> dict:
    """Get an empty hotel-address fixture."""
    return {}


def _namespace() -> ns.Namespace:
    runtime = functions_runtime.FunctionsRuntime()
    runtime.register_function(get_restaurants_address)
    runtime.register_function(get_rating_reviews_for_restaurants)
    runtime.register_function(get_current_day)
    runtime.register_function(get_iban)
    runtime.register_function(reserve_restaurant)
    runtime.register_function(read_file)
    runtime.register_function(get_user_info)
    runtime.register_function(get_hotels_address)
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


@pytest.mark.parametrize(
    ("tool", "expression", "expected_raw", "inner_sources", "expected_readers"),
    (
        (
            "get_current_day",
            "get_current_day()",
            "M",
            frozenset({sources.SourceEnum.TrustedToolSource}),
            readers.Public(),
        ),
        (
            "reserve_restaurant",
            "reserve_restaurant()",
            "O",
            frozenset({sources.SourceEnum.TrustedToolSource}),
            frozenset(),
        ),
        ("read_file", "read_file()", "s", frozenset(), frozenset()),
    ),
)
def test_top_level_string_indexing_exposes_tool_metadata(
    tool: str,
    expression: str,
    expected_raw: str,
    inner_sources: frozenset[sources.SourceEnum],
    expected_readers: readers.Readers,
):
    tool_source = sources.Tool(tool, inner_sources)
    expected_metadata = Capabilities(frozenset({tool_source}), expected_readers)

    ev = _eval(f"out = {expression}\nselected = out[0]", _namespace())
    selected = _bound(ev, "selected")

    assert isinstance(selected, value._CaMeLChar)
    assert selected.raw == expected_raw
    assert selected.metadata == expected_metadata
    _assert_effective_metadata(
        selected,
        frozenset({tool_source, sources.SourceEnum.CaMeL, sources.SourceEnum.User}),
        expected_readers,
    )


def test_top_level_string_iteration_exposes_character_metadata():
    tool = "get_current_day"
    tool_source = sources.Tool(tool, frozenset({sources.SourceEnum.TrustedToolSource}))
    expected_metadata = _tool_metadata(
        tool,
        frozenset({sources.SourceEnum.TrustedToolSource}),
        readers.Public(),
    )

    ev = _eval("out = get_current_day()\nfor item in out:\n    selected = item", _namespace())
    selected = _bound(ev, "selected")

    assert isinstance(selected, value._CaMeLChar)
    assert selected.raw == "y"
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


@pytest.mark.parametrize(
    ("expression", "wrapper_type", "expected_raw", "tool"),
    (
        ("get_iban()", value.CaMeLStr, "", "get_iban"),
        ("get_hotels_address()", value.CaMeLDict, {}, "get_hotels_address"),
    ),
)
def test_empty_tool_outputs_keep_root_metadata(expression, wrapper_type, expected_raw, tool: str):
    expected_metadata = _tool_metadata(
        tool,
        frozenset({sources.SourceEnum.TrustedToolSource}),
        readers.Public(),
    )

    ev = _eval(f"out = {expression}", _namespace())
    output = _bound(ev, "out")

    assert isinstance(output, wrapper_type)
    assert output.raw == expected_raw
    assert output.metadata == expected_metadata


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


@pytest.mark.parametrize("container_type", (value.CaMeLList, value.CaMeLTuple, value.CaMeLSet))
def test_top_level_sequence_members_use_tool_specific_metadata(container_type):
    tool = "get_restaurants_address"
    metadata = Capabilities.default()
    leaf = value.CaMeLStr.from_raw("123 Main St", metadata, ())
    wrapped = container_type([leaf], metadata, ())

    converted = _get_metadata_for_ad(wrapped, tool)
    converted_leaf = next(iter(converted._python_value))
    expected_leaf_metadata = _tool_metadata(
        tool,
        frozenset({sources.SourceEnum.TrustedToolSource}),
        readers.Public(),
    )
    expected_root_metadata = _tool_metadata(tool, frozenset(), readers.Public())
    assert converted.raw == wrapped.raw
    assert type(converted) is container_type
    assert converted.metadata == expected_root_metadata
    assert converted_leaf.metadata == expected_leaf_metadata
    assert all(character.metadata == expected_leaf_metadata for character in converted_leaf._python_value)
    assert wrapped.metadata == metadata
    assert leaf.metadata == metadata


def test_top_level_string_characters_receive_boundary_metadata():
    tool = "get_restaurants_address"
    wrapped = value.CaMeLStr.from_raw("address", Capabilities.default(), ())

    converted = _get_metadata_for_ad(wrapped, tool)
    expected_metadata = _tool_metadata(
        tool,
        frozenset({sources.SourceEnum.TrustedToolSource}),
        readers.Public(),
    )
    assert converted.raw == wrapped.raw
    assert converted.metadata == expected_metadata
    assert all(character.metadata == expected_metadata for character in converted._python_value)
    assert wrapped.metadata == Capabilities.default()


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
