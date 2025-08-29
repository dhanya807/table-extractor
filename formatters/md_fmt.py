from __future__ import annotations

import os
from typing import Dict, List

from tabulate import tabulate


def _table_to_markdown(tbl: Dict, page_number: int) -> str:
    headers = tbl.get("headers", [])
    rows = tbl.get("rows", [])

    md = ""

    # Flatten all row groups into a single list if nested
    if rows and isinstance(rows[0], list) and rows and isinstance(rows[0][0], list):
        # Merge multi-line item groups: join description-related lines beneath main row
        merged_rows: List[List[str]] = []
        for group in rows:
            if not group:
                continue
            main = group[0][:]
            # Find description column index heuristically
            desc_idx = None
            for i, h in enumerate(headers):
                if str(h).strip().lower() in {"description"}:
                    desc_idx = i
                    break
            # Append continuation lines into description
            extras: List[str] = []
            for cont in group[1:]:
                parts = [p for p in cont if str(p).strip()]
                if not parts:
                    continue
                text = " ".join(parts)
                extras.append(text)
            if desc_idx is not None and extras:
                base = (main[desc_idx] or "").strip()
                extra_text = " | ".join(extras)
                main[desc_idx] = (base + (" | " if base and extra_text else "") + extra_text).strip()
            merged_rows.append(main)
        md += tabulate(merged_rows, headers=headers, tablefmt="github") + "\n\n"
    else:
        # Traditional flat structure
        md += tabulate(rows, headers=headers, tablefmt="github") + "\n\n"

    return md


def to_markdown_file(output_path: str, file_path: str, page_number: int, tables: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunks: List[str] = []
    if tables and isinstance(tables[0], dict):
        header = tables[0].get("page_header")
        if header and isinstance(header, dict):
            title = header.get("title") or (header.get("lines") or [None])[0]
            date = header.get("date")
            time_val = header.get("time")
            invoice_no = header.get("invoice_no")
            if title:
                chunks.append(f"# {title}\n")
            meta_line = []
            if date:
                meta_line.append(f"Dated: {date}")
            if time_val:
                meta_line.append(f"Time: {time_val}")
            if meta_line:
                chunks.append("  ".join(meta_line) + "\n")
            if invoice_no:
                chunks.append(f"Invoice No: {invoice_no}\n\n")

    chunks.extend([_table_to_markdown(t, page_number) for t in tables])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(chunks))


def to_html_file(output_path: str, file_path: str, page_number: int, tables: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Tables</title></head><body>"
    ]
    if tables and isinstance(tables[0], dict):
        header = tables[0].get("page_header")
        if header and isinstance(header, dict):
            lines = header.get("lines") or []
            if lines:
                html_parts.append(f"<h1>{lines[0]}</h1>")

    for t in tables:
        headers = t.get("headers", [])
        rows = t.get("rows", [])

        # Flatten all row groups into a single list if nested
        if rows and isinstance(rows[0], list) and rows and isinstance(rows[0][0], list):
            flat_rows = [row for group in rows for row in group]
            html_parts.append("<table border='1' cellspacing='0' cellpadding='4'>")
            if headers:
                html_parts.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
            html_parts.append("<tbody>")
            for r in flat_rows:
                html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            html_parts.append("</tbody></table>")
        else:
            html_parts.append("<table border='1' cellspacing='0' cellpadding='4'>")
            if headers:
                html_parts.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>")
            html_parts.append("<tbody>")
            for r in rows:
                html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            html_parts.append("</tbody></table>")

    html_parts.append("</body></html>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))