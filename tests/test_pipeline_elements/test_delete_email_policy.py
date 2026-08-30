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

"""Tests that `delete_email_policy` allows trusted user-direct IDs and denies
untrusted tool-derived IDs.
"""

import agentdojo.task_suite  # noqa: F401  (import-order workaround)

from camel import security_policy
from camel.capabilities import Capabilities, readers, sources
from camel.interpreter import value
from camel.pipeline_elements.security_policies.workspace import WorkspaceSecurityPolicyEngine


def _trusted(s: str) -> value.CaMeLValue:
    return value.CaMeLStr.from_raw(s, Capabilities.default(), ())


def _untrusted(s: str) -> value.CaMeLValue:
    return value.CaMeLStr.from_raw(
        s,
        Capabilities(frozenset({sources.Tool("get_unread_emails", frozenset())}), readers.Public()),
        (),
    )


def test_user_direct_email_id_is_allowed():
    engine = WorkspaceSecurityPolicyEngine(None)  # type: ignore[arg-type]
    verdict = engine.delete_email_policy("delete_email", {"email_id": _trusted("42")})
    assert isinstance(verdict, security_policy.Allowed)


def test_tool_derived_email_id_is_denied():
    engine = WorkspaceSecurityPolicyEngine(None)  # type: ignore[arg-type]
    verdict = engine.delete_email_policy("delete_email", {"email_id": _untrusted("42")})
    assert isinstance(verdict, security_policy.Denied)
