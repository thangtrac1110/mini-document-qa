"""Tokenizer shared by indexing and querying, so both sides always match.

Tokens are maximal runs of Unicode word characters (letters, digits, underscore) after
NFC normalization and case folding. Diacritics are kept, so "dấu" and "dâu" stay distinct.

Limitation: for Vietnamese this yields *syllables*, not words ("học sinh" -> ["học", "sinh"]).
Proper word segmentation would need a dedicated tool and is out of scope for the MVP.
No stemming and no stopword removal: BM25's IDF already down-weights very common terms.
"""

import re
import unicodedata

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFC", text).casefold())
