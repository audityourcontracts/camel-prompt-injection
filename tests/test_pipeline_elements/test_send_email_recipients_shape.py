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

"""Tests recipient-shape validation for primary, CC, and BCC email recipients.

All recipient elements must be strings, including in trusted aggregates.
"""

import agentdojo.task_suite  # noqa: F401  (import-order workaround)
import pytest

from camel import security_policy
from camel.capabilities import Capabilities, readers, sources
from camel.interpreter import value
from camel.pipeline_elements.security_policies.travel import TravelSecurityPolicyEngine
from camel.pipeline_elements.security_policies.workspace import WorkspaceSecurityPolicyEngine

_ENGINES = [WorkspaceSecurityPolicyEngine, TravelSecurityPolicyEngine]


def _trusted(s: str) -> value.CaMeLValue:
    """User-direct (trusted) string value."""
    return value.CaMeLStr.from_raw(s, Capabilities.default(), ())


def _untrusted(s: str) -> value.CaMeLValue:
    """Tool-derived (untrusted) string value."""
    return value.CaMeLStr.from_raw(
        s,
        Capabilities(frozenset({sources.Tool("get_unread_emails", frozenset())}), readers.Public()),
        (),
    )


def _a_dict() -> value.CaMeLValue:
    """A (trusted) dict — not a valid recipient."""
    return value.CaMeLDict(
        {value.CaMeLStr.from_raw("k", Capabilities.default(), ()): _trusted("v")},
        Capabilities.default(),
        (),
    )


def _malformed_untrusted_recipients() -> value.CaMeLValue:
    """Untrusted tool-derived aggregate containing an invalid dict recipient."""
    return value.CaMeLList(
        [_untrusted("attacker@evil.example"), _a_dict()],
        Capabilities(frozenset({sources.Tool("get_unread_emails", frozenset())}), readers.Public()),
        (),
    )


def _malformed_trusted_recipients() -> value.CaMeLValue:
    """Trusted user-direct aggregate containing an invalid dict recipient."""
    return value.CaMeLList([_trusted("ok@example.com"), _a_dict()], Capabilities.default(), ())


def _kwargs(recipients: value.CaMeLValue, **extra) -> dict:
    return {
        "recipients": recipients,
        "subject": _trusted("hello"),
        "body": _trusted("see attached"),
        **extra,
    }


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_malformed_untrusted_recipients_are_denied(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    verdict = engine.send_email_policy("send_email", _kwargs(_malformed_untrusted_recipients()))
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_malformed_trusted_primary_recipients_are_denied(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    verdict = engine.send_email_policy("send_email", _kwargs(_malformed_trusted_recipients()))
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_malformed_cc_recipients_are_denied(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    cc = value.CaMeLList([_a_dict()], Capabilities.default(), ())
    verdict = engine.send_email_policy(
        "send_email", _kwargs(value.CaMeLList([_trusted("ok@example.com")], Capabilities.default(), ()), cc=cc)
    )
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_malformed_bcc_recipients_are_denied(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    bcc = value.CaMeLList([_a_dict()], Capabilities.default(), ())
    verdict = engine.send_email_policy(
        "send_email", _kwargs(value.CaMeLList([_trusted("ok@example.com")], Capabilities.default(), ()), bcc=bcc)
    )
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_well_formed_trusted_recipients_are_allowed(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    recipients = value.CaMeLList([_trusted("ok@example.com")], Capabilities.default(), ())
    verdict = engine.send_email_policy("send_email", _kwargs(recipients))
    assert isinstance(verdict, security_policy.Allowed)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_well_formed_untrusted_recipients_are_allowed(engine_cls):
    """Well-formed untrusted recipients are allowed with public content."""
    engine = engine_cls(None)  # type: ignore[arg-type]
    recipients = value.CaMeLList(
        [_untrusted("attacker@evil.example")],
        Capabilities(frozenset({sources.Tool("get_unread_emails", frozenset())}), readers.Public()),
        (),
    )
    verdict = engine.send_email_policy("send_email", _kwargs(recipients))
    assert isinstance(verdict, security_policy.Allowed)
