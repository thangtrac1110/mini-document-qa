from app.retrieval.tokenizer import tokenize


def test_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World! BM25-ranking.") == ["hello", "world", "bm25", "ranking"]


def test_keeps_vietnamese_diacritics_as_distinct_syllables():
    assert tokenize("Học sinh đọc SÁCH") == ["học", "sinh", "đọc", "sách"]
    assert tokenize("dấu") != tokenize("dâu")


def test_decomposed_and_composed_unicode_match():
    assert tokenize("Việt") == tokenize("Việt")


def test_punctuation_only_gives_no_tokens():
    assert tokenize(" ?!... -- ") == []
