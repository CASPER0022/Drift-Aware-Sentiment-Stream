"""Tweet cleaning and tokenisation shared by the offline runner and the Spark consumer.

Both paths must call `tokenize` so that the model sees identical features no matter
how the tweet reached it. Rules (timeline section 3.4): lowercase, unescape HTML
entities, drop URLs and @mentions, keep hashtag words without '#', squeeze characters
repeated 3+ times down to 2, split into unigrams. No stemming or stop-word removal.
"""
import html
import re

_URL = re.compile(r"(?:https?://|www\.)\S+")
_MENTION = re.compile(r"@\w+")
_REPEAT = re.compile(r"(.)\1{2,}")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")  # keeps contractions like "don't", "i'm"


def clean(text: str) -> str:
    text = html.unescape(text).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", " ")
    return _REPEAT.sub(r"\1\1", text)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(clean(text))
