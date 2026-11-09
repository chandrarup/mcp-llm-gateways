"""Streaming PII redaction with a bounded hold-back buffer.

A PII value can be split across stream chunks ("john.do" | "e@exa" | "mple.com"), so text
can only be released once nothing at its end could still turn into a match. The redactor
keeps a small tail buffer:

  * the trailing non-whitespace run (an email or an unseparated card/SSN may still be growing)
  * the trailing run of digits/spaces/dashes (a card like "4111 1111 1111 1111" spans spaces)
  * any complete match that touches the end of the buffer ("a@b.co" may become "a@b.com")

Everything before that tail is redacted and emitted immediately, so ordinary prose streams
with at most one partial word of delay. MAX_BUFFER bounds memory if a client streams an
endless token without whitespace.
"""

import re

REDACTED = "[REDACTED]"
MAX_BUFFER = 512

PII_PATTERN = re.compile(
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
    r"|(?P<ssn>(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9]))"
    r"|(?P<card>(?<![0-9])[0-9](?:[ -]?[0-9]){12,18}(?![0-9]))"
)
_TRAILING_TOKEN = re.compile(r"\S*\Z")
_TRAILING_DIGIT_RUN = re.compile(r"[0-9][0-9 -]*\Z")


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        checksum += d
    return checksum % 10 == 0


def _is_pii(match: re.Match[str]) -> bool:
    return match.lastgroup != "card" or luhn_valid(match.group())


def redact_text(text: str) -> str:
    """Redact a complete string (non-streaming path, and the reference for the streaming one)."""
    return PII_PATTERN.sub(lambda m: REDACTED if _is_pii(m) else m.group(), text)


class StreamRedactor:
    def __init__(self, max_buffer: int = MAX_BUFFER) -> None:
        self._buf = ""
        self._max_buffer = max_buffer

    def feed(self, chunk: str) -> str:
        """Add a chunk; return the text that is now safe to send."""
        self._buf += chunk
        return self._drain(final=False)

    def flush(self) -> str:
        """End of stream: redact and release everything still held."""
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        buf = self._buf
        if final:
            cut = len(buf)
        else:
            cut = min(_TRAILING_TOKEN.search(buf).start(), len(buf))  # type: ignore[union-attr]
            digit_run = _TRAILING_DIGIT_RUN.search(buf)
            if digit_run:
                cut = min(cut, digit_run.start())

        out: list[str] = []
        pos = 0
        for m in PII_PATTERN.finditer(buf):
            if m.end() > cut or (not final and m.end() == len(buf)):
                cut = min(cut, m.start())
                break
            out.append(buf[pos : m.start()])
            out.append(REDACTED if _is_pii(m) else m.group())
            pos = m.end()

        # Bound memory: if the held tail is too long, force out its oldest part (still redacted).
        if len(buf) - cut > self._max_buffer:
            cut = len(buf) - self._max_buffer
        cut = max(cut, pos)

        out.append(redact_text(buf[pos:cut]))
        self._buf = buf[cut:]
        return "".join(out)
