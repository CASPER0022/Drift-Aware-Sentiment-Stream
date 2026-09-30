from core.preprocess import clean, tokenize


def test_lowercases_and_splits():
    assert tokenize("Going to BED now.") == ["going", "to", "bed", "now"]


def test_drops_urls_and_mentions():
    text = "@swiv see -http://dragtotop.com/x and www.imnotokay.net ok"
    assert tokenize(text) == ["see", "and", "ok"]


def test_keeps_hashtag_word():
    assert tokenize("loving #CampRock") == ["loving", "camprock"]


def test_unescapes_html_entities():
    assert tokenize("lost my voice &gt;.&lt; &amp; more") == ["lost", "my", "voice", "more"]


def test_squeezes_repeated_characters():
    assert clean("sooooo goooood!!!") == "soo good!!"
    assert tokenize("Okayyy?!?!?") == ["okayy"]


def test_keeps_contractions():
    assert tokenize("I can't, don't ask") == ["i", "can't", "don't", "ask"]


def test_empty_after_cleaning():
    assert tokenize("@someone http://x.y") == []
