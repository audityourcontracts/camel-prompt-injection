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

import ast
import dataclasses
from types import SimpleNamespace

import agentdojo.task_suite  # noqa: F401  (import-order workaround)
import pytest
from agentdojo import functions_runtime

from camel import security_policy
from camel.capabilities import Capabilities, readers, sources
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.pipeline_elements import agentdojo_function
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
    verdict = engine.check_policy("send_email", _kwargs(_malformed_untrusted_recipients()), ())
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_malformed_trusted_primary_recipients_are_denied(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    verdict = engine.check_policy("send_email", _kwargs(_malformed_trusted_recipients()), ())
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_malformed_cc_recipients_are_denied(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    cc = value.CaMeLList([_a_dict()], Capabilities.default(), ())
    verdict = engine.check_policy(
        "send_email", _kwargs(value.CaMeLList([_trusted("ok@example.com")], Capabilities.default(), ()), cc=cc), ()
    )
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_malformed_bcc_recipients_are_denied(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    bcc = value.CaMeLList([_a_dict()], Capabilities.default(), ())
    verdict = engine.check_policy(
        "send_email", _kwargs(value.CaMeLList([_trusted("ok@example.com")], Capabilities.default(), ()), bcc=bcc), ()
    )
    assert isinstance(verdict, security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
def test_well_formed_trusted_recipients_are_allowed(engine_cls):
    engine = engine_cls(None)  # type: ignore[arg-type]
    recipients = value.CaMeLList([_trusted("ok@example.com")], Capabilities.default(), ())
    verdict = engine.check_policy("send_email", _kwargs(recipients), ())
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
    verdict = engine.check_policy("send_email", _kwargs(recipients), ())
    assert isinstance(verdict, security_policy.Allowed)


def _list(*items: value.CaMeLValue) -> value.CaMeLList:
    return value.CaMeLList(items, Capabilities.default(), ())


def _restricted(text: str, allowed_readers: frozenset[str]) -> value.CaMeLStr:
    return value.CaMeLStr.from_raw(text, Capabilities(frozenset({sources.SourceEnum.User}), allowed_readers), ())


def _check(engine_cls, kwargs, env=None):
    return engine_cls(env).check_policy("send_email", kwargs, ())


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("channel", ("recipients", "cc", "bcc"))
@pytest.mark.parametrize("kind", ("tuple", "set", "str", "dict"))
def test_wrong_recipient_outer_shape_is_denied(engine_cls, channel, kind):
    recipient = _trusted("optional@example.com")
    malformed = {
        "tuple": value.CaMeLTuple([recipient], Capabilities.default(), ()),
        "set": value.CaMeLSet([recipient], Capabilities.default(), ()),
        "str": recipient,
        "dict": _a_dict(),
    }[kind]
    kwargs = _kwargs(_list(_trusted("primary@example.com")))
    kwargs[channel] = malformed
    assert isinstance(_check(engine_cls, kwargs), security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("missing", (False, True))
def test_primary_none_or_missing_is_denied(engine_cls, missing):
    kwargs = _kwargs(value.CaMeLNone(Capabilities.default(), ()))
    if missing:
        del kwargs["recipients"]
    assert isinstance(_check(engine_cls, kwargs), security_policy.Denied)


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("none_channels", ((), ("cc",), ("bcc",), ("cc", "bcc")))
def test_optional_none_and_absence_are_allowed(engine_cls, none_channels):
    kwargs = _kwargs(_list(_trusted("primary@example.com")))
    kwargs.update({channel: value.CaMeLNone(Capabilities.default(), ()) for channel in none_channels})
    assert isinstance(_check(engine_cls, kwargs), security_policy.Allowed)


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("channels", (("cc",), ("bcc",), ("cc", "bcc")))
@pytest.mark.parametrize("content", ("public", "shared", "restricted"))
def test_optional_lists_participate_in_body_reader_checks(engine_cls, channels, content):
    kwargs = _kwargs(_list(_trusted("primary@example.com")))
    kwargs.update({channel: _list(_untrusted(f"{channel}@example.com")) for channel in channels})
    identities = {"primary@example.com", *(f"{channel}@example.com" for channel in channels)}
    if content != "public":
        allowed_readers = identities if content == "shared" else {"primary@example.com"}
        kwargs["body"] = _restricted("body", frozenset(allowed_readers))
    originals = dict(kwargs)
    expected = security_policy.Denied if content == "restricted" else security_policy.Allowed
    assert isinstance(_check(engine_cls, kwargs), expected)
    assert all(kwargs[name] is original for name, original in originals.items())
    assert {
        recipient.raw for channel in ("recipients", *channels) for recipient in kwargs[channel].iterate_python()
    } == identities


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("channel", ("recipients", "cc", "bcc"))
@pytest.mark.parametrize("untrusted_part", ("container", "member", "neither"))
def test_trust_shortcut_requires_container_and_member_trust(engine_cls, channel, untrusted_part):
    recipient = _untrusted("other@example.com") if untrusted_part == "member" else _trusted("other@example.com")
    recipients = _list(recipient)
    if untrusted_part == "container":
        recipients = recipients.new_with_metadata(
            Capabilities(frozenset({sources.Tool("get_unread_emails")}), readers.Public())
        )
    kwargs = _kwargs(_list(_trusted("primary@example.com")), body=_restricted("private", frozenset()))
    kwargs[channel] = recipients
    expected = security_policy.Allowed if untrusted_part == "neither" else security_policy.Denied
    assert isinstance(_check(engine_cls, kwargs), expected)


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("field", ("subject", "attachments"))
@pytest.mark.parametrize("shared", (False, True))
def test_optional_recipients_participate_in_other_content_reader_checks(engine_cls, field, shared):
    identities = {"primary@example.com", "optional@example.com"}
    allowed_readers = frozenset(identities if shared else {"primary@example.com"})
    kwargs = _kwargs(_list(_trusted("primary@example.com")), bcc=_list(_untrusted("optional@example.com")))
    env = None
    if field == "subject":
        kwargs[field] = _restricted("subject", allowed_readers)
    else:
        attachment = value.CaMeLDict(
            {_trusted("type"): _trusted("file"), _trusted("file_id"): _trusted("file-1")},
            Capabilities.default(),
            (),
        )
        kwargs[field] = _list(attachment).new_with_metadata(
            Capabilities(frozenset({sources.SourceEnum.User}), allowed_readers)
        )
        file = SimpleNamespace(id_="file-1", shared_with=dict.fromkeys(identities))
        env = SimpleNamespace(cloud_drive=SimpleNamespace(files={"file-1": file}))
    expected = security_policy.Allowed if shared else security_policy.Denied
    assert isinstance(_check(engine_cls, kwargs, env), expected)


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("empty_primary", (False, True))
@pytest.mark.parametrize("optional_member", (False, True))
def test_empty_recipient_lists_preserve_existing_behavior(engine_cls, empty_primary, optional_member):
    primary = _list() if empty_primary else _list(_trusted("primary@example.com"))
    cc = _list(_untrusted("optional@example.com")) if optional_member else _list()
    assert isinstance(_check(engine_cls, _kwargs(primary, cc=cc, bcc=_list())), security_policy.Allowed)


@pytest.mark.parametrize("engine_cls", _ENGINES)
@pytest.mark.parametrize("malformed", (False, True))
def test_interpreter_dispatches_only_valid_recipient_channels(engine_cls, malformed):
    calls = []

    def send_email(
        recipients: list[str], subject: str, body: str, cc: list[str] | None = None, bcc: list[str] | None = None
    ) -> None:
        """Record email arguments locally without delivering email.

        :param recipients: Primary recipients.
        :param subject: Email subject.
        :param body: Email body.
        :param cc: Carbon-copy recipients.
        :param bcc: Blind-carbon-copy recipients.
        """
        calls.append({"recipients": recipients, "subject": subject, "body": body, "cc": cc, "bcc": bcc})

    runtime = functions_runtime.FunctionsRuntime()
    runtime.register_function(send_email)
    namespace = ns.Namespace.with_builtins()
    namespace = dataclasses.replace(
        namespace,
        variables=namespace.variables | agentdojo_function.make_agentdojo_namespace(namespace, runtime, None),
    )
    cc = "('cc@example.com',)" if malformed else "['cc@example.com']"
    code = ast.parse(
        f"send_email(recipients=['primary@example.com'], subject='subject', body='body', cc={cc}, bcc=['bcc@example.com'])"
    )
    eval_args = interpreter.EvalArgs(engine_cls(None), interpreter.MetadataEvalMode.NORMAL)
    if malformed:
        with pytest.raises(security_policy.SecurityPolicyDeniedError):
            interpreter.camel_eval(code, namespace, [], [], eval_args)
        assert calls == []
    else:
        evaluation = interpreter.camel_eval(code, namespace, [], [], eval_args)
        assert isinstance(evaluation.result, result.Ok)
        assert calls == [
            {
                "recipients": ["primary@example.com"],
                "subject": "subject",
                "body": "body",
                "cc": ["cc@example.com"],
                "bcc": ["bcc@example.com"],
            }
        ]
