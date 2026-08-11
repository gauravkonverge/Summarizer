from app.services.text_quality import assess_garbled


def test_flags_repeated_symbol_noise():
    is_garbled, reason = assess_garbled("@@@@ #### NULL ERROR")
    assert is_garbled
    assert reason


def test_flags_repeated_percent_run():
    is_garbled, reason = assess_garbled("%%%% unreadable payload")
    assert is_garbled
    assert reason


def test_flags_partial_corruption_with_real_words():
    is_garbled, reason = assess_garbled("vehicle ???? status ????")
    assert is_garbled
    assert reason


def test_flags_unicode_replacement_characters():
    is_garbled, reason = assess_garbled("��� invoice total")
    assert is_garbled
    assert "encoding failure" in reason


def test_allows_normal_message():
    is_garbled, reason = assess_garbled("We will provide an order update.")
    assert not is_garbled
    assert reason is None


def test_allows_ordinary_emphasis_punctuation():
    is_garbled, _ = assess_garbled("Wait!!! That's great, right?")
    assert not is_garbled


def test_allows_error_codes_and_percentages():
    is_garbled, _ = assess_garbled("Error E-042 (check engine): 100% confirmed, call (555) 123-4567.")
    assert not is_garbled


def test_allows_urls():
    is_garbled, _ = assess_garbled("Check https://example.com/status for the vehicle status.")
    assert not is_garbled


def test_flags_short_symbol_only_message():
    is_garbled, reason = assess_garbled("???")
    assert is_garbled
    assert reason
