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

"""Tests that travel reservation policies validate each tool's declared fields.

`reserve_car_rental` validates `company`, `start_time`, and `end_time`;
`reserve_restaurant` validates `restaurant` and `start_time`.
"""

import agentdojo.task_suite  # noqa: F401  (import-order workaround)

from camel import security_policy
from camel.capabilities import Capabilities, readers, sources
from camel.interpreter import value
from camel.pipeline_elements.security_policies.travel import TravelSecurityPolicyEngine


def _trusted(s: str) -> value.CaMeLValue:
    return value.CaMeLStr.from_raw(s, Capabilities.default(), ())


def _untrusted(s: str) -> value.CaMeLValue:
    return value.CaMeLStr.from_raw(
        s,
        Capabilities(frozenset({sources.Tool("get_rating_reviews_for_car_rental", frozenset())}), readers.Public()),
        (),
    )


def test_legitimate_car_rental_call_is_allowed():
    engine = TravelSecurityPolicyEngine(None)  # type: ignore[arg-type]
    verdict = engine.check_policy(
        "reserve_car_rental",
        {"company": _trusted("Hertz"), "start_time": _trusted("t1"), "end_time": _trusted("t2")},
        (),
    )
    assert isinstance(verdict, security_policy.Allowed)


def test_legitimate_restaurant_call_is_allowed():
    engine = TravelSecurityPolicyEngine(None)  # type: ignore[arg-type]
    verdict = engine.check_policy(
        "reserve_restaurant",
        {"restaurant": _trusted("Olive"), "start_time": _trusted("t1")},
        (),
    )
    assert isinstance(verdict, security_policy.Allowed)


def test_untrusted_company_is_denied():
    engine = TravelSecurityPolicyEngine(None)  # type: ignore[arg-type]
    verdict = engine.check_policy(
        "reserve_car_rental",
        {"company": _untrusted("Evil Rentals"), "start_time": _trusted("t1"), "end_time": _trusted("t2")},
        (),
    )
    assert isinstance(verdict, security_policy.Denied)


def test_extra_restaurant_kwarg_does_not_bypass_company_check():
    """An extra trusted `restaurant` kwarg does not bypass validation of the
    untrusted `company` field.
    """
    engine = TravelSecurityPolicyEngine(None)  # type: ignore[arg-type]
    verdict = engine.check_policy(
        "reserve_car_rental",
        {
            "company": _untrusted("Evil Rentals"),
            "start_time": _trusted("t1"),
            "end_time": _trusted("t2"),
            "restaurant": _trusted("decoy"),
        },
        (),
    )
    assert isinstance(verdict, security_policy.Denied)
