from extractor.merger import normalize_table, merge_multi_page_tables


def test_normalize_table():
    t = {"title": None, "headers": [" A ", "B"], "rows": [[" x ", " y "]]}
    n = normalize_table(t)
    assert n["headers"] == ["A", "B"]
    assert n["rows"] == [["x", "y"]]


def test_merge_multi_page_tables_should_merge():
    a = [{"title": "T1", "headers": ["h1", "h2"], "rows": [["1", "2"]]}]
    b = [{"title": None, "headers": [], "rows": [["3", "4"]]}]  # No headers/title, should merge
    merged = merge_multi_page_tables([a, b])
    assert len(merged) == 1
    assert merged[0]["rows"] == [["1", "2"], ["3", "4"]]


def test_merge_multi_page_tables_should_not_merge():
    a = [{"title": "T1", "headers": ["h1", "h2"], "rows": [["1", "2"]]}]
    b = [{"title": "T2", "headers": ["h1", "h2"], "rows": [["3", "4"]]}]  # Has title/headers, should not merge
    merged = merge_multi_page_tables([a, b])
    assert len(merged) == 2
    assert merged[0]["rows"] == [["1", "2"]]
    assert merged[1]["rows"] == [["3", "4"]]