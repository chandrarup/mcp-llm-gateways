import pytest

from task3_llm_guardrail.redactor import REDACTED, StreamRedactor, luhn_valid, redact_text

SAMPLES = [
    "Contact john.doe@example.com for details.",
    "SSN 123-45-6789 on file.",
    "Card 4111 1111 1111 1111 expires soon.",
    "Card 4111-1111-1111-1111 and 5500005555555559 both.",
    "Email a@b.co then a@b.com, SSN 078-05-1120, card 378282246310005.",
    "No PII here, just 3 apples and 42 oranges in 2026.",
    "Order 1234567890123 is not a card (fails Luhn).",
    "ends with an email jane@corp.io",
    "ends with digits 4111 1111 1111 1111",
]


def stream(chunks: list[str]) -> str:
    r = StreamRedactor()
    return "".join(r.feed(c) for c in chunks) + r.flush()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mail john.doe@example.com now", f"mail {REDACTED} now"),
        ("ssn 123-45-6789.", f"ssn {REDACTED}."),
        ("card 4111 1111 1111 1111!", f"card {REDACTED}!"),
        ("card 4111111111111111", f"card {REDACTED}"),
        ("not a card 1234567890123", "not a card 1234567890123"),
        ("phone 555-123-4567", "phone 555-123-4567"),
        ("year 2026 and 1234-56-789", "year 2026 and 1234-56-789"),
    ],
)
def test_redact_text(text, expected):
    assert redact_text(text) == expected


def test_luhn():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")


@pytest.mark.parametrize("text", SAMPLES)
def test_every_two_way_split_matches_whole_text_redaction(text):
    expected = redact_text(text)
    for i in range(len(text) + 1):
        assert stream([text[:i], text[i:]]) == expected, f"split at {i}: {text[:i]!r} | {text[i:]!r}"


@pytest.mark.parametrize("text", SAMPLES)
def test_every_three_way_split_matches_whole_text_redaction(text):
    expected = redact_text(text)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            assert stream([text[:i], text[i:j], text[j:]]) == expected


@pytest.mark.parametrize("text", SAMPLES)
def test_character_by_character_stream(text):
    assert stream(list(text)) == redact_text(text)


def test_raw_pii_never_appears_in_any_emitted_piece():
    r = StreamRedactor()
    pieces = [r.feed(c) for c in "reach me at john.doe@example.com today"] + [r.flush()]
    joined = "".join(pieces)
    assert "john" not in joined and "example" not in joined


def test_prose_is_released_immediately_for_low_ttft():
    r = StreamRedactor()
    assert r.feed("Hello there, how ") == "Hello there, how "
    assert r.feed("are yo") == "are "
    assert r.feed("u?") == ""
    assert r.flush() == "you?"


def test_digits_are_held_until_they_cannot_be_a_card():
    r = StreamRedactor()
    assert r.feed("pay 4111 ") == "pay "
    assert r.feed("1111 1111 1111 ") == ""
    assert r.feed("and more") == f"{REDACTED} and "


def test_buffer_is_bounded_for_endless_tokens():
    r = StreamRedactor(max_buffer=64)
    emitted = "".join(r.feed("x" * 10) for _ in range(100))
    assert len(r._buf) <= 64
    assert len(emitted) + len(r.flush()) == 1000
