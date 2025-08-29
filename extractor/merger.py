from __future__ import annotations

from typing import Dict, List

import pandas as pd


def normalize_table(table: Dict) -> Dict:
    headers = [str(h).strip() for h in table.get("headers", [])]
    rows = table.get("rows", [])
    if rows and isinstance(rows[0], list) and isinstance(rows[0][0], list):
        flattened_rows = []
        for row_group in rows:
            for row in row_group:
                flattened_rows.append([str(c).strip() for c in row])
        rows = flattened_rows
    else:
        rows = [[str(c).strip() for c in r] for r in rows]
    return {"title": table.get("title"), "headers": headers, "rows": rows}


def _can_concatenate(a: Dict, b: Dict) -> bool:
    # Only merge if b has no headers or title (i.e., it's a continuation)
    b_headers = b.get("headers", [])
    b_title = b.get("title")
    a_headers = a.get("headers", [])
    # b_headers must be empty or match a_headers, and b_title must be None or empty
    return (not b_headers or b_headers == a_headers) and (not b_title)


def merge_multi_page_tables(tables_by_page: List[List[Dict]]) -> List[Dict]:
    merged: List[Dict] = []
    buffer: Dict | None = None

    for page_tables in tables_by_page:
        for tbl in page_tables:
            tbl = normalize_table(tbl)
            if buffer is None:
                buffer = tbl
                continue
            if _can_concatenate(buffer, tbl):
                # Merge rows only, keep buffer's headers/title
                buffer["rows"].extend(tbl["rows"])
            else:
                merged.append(buffer)
                buffer = tbl
    if buffer is not None:
        merged.append(buffer)
    return merged