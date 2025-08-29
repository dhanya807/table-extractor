from __future__ import annotations

from typing import Dict, List

import camelot
import pandas as pd
from .pdf_utils import extract_page_text
import re


def _detect_table_structure(df: pd.DataFrame) -> tuple[bool, int, List[str]]:
    """Automatically detect if this is a structured table and find the header row."""
    
    # Look for rows that could be headers (contain multiple non-empty, non-numeric values)
    potential_headers = []
    
    for i, row in df.iterrows():
        row_values = [str(c).strip() for c in row if pd.notna(c) and str(c).strip()]
        if len(row_values) >= 3:  # At least 3 columns
            # Check if this row looks like a header (not mostly numbers, has meaningful text)
            numeric_count = sum(1 for v in row_values if v.replace('.', '').replace(',', '').replace('-', '').isdigit())
            if numeric_count < len(row_values) * 0.3:  # Less than 30% numbers
                potential_headers.append((i, row_values))
    
    if not potential_headers:
        return False, -1, []
    
    # Find the best header row (usually the one with the most distinct values)
    best_header_idx = -1
    best_header_score = 0
    best_headers = []
    
    generic_header_keywords = {
        "carton", "order", "order no", "orderno", "part", "part number",
        "description", "unit", "unit price", "qty", "quantity", "sales",
        "value", "customs", "price"
    }
    bad_header_keywords = {"dated", "time", "email", "e-mail", "tel", "fax"}

    for idx, headers in potential_headers:
        # Score based on uniqueness and length
        normalized = [h.lower() for h in headers]
        unique_headers = len(set(normalized))
        avg_length = sum(len(h) for h in headers) / len(headers)

        # Boost if this looks like real table headers
        joined = " ".join(normalized)
        good_hits = sum(1 for kw in generic_header_keywords if kw in joined)
        bad_hits = sum(1 for kw in bad_header_keywords if kw in joined)

        score = unique_headers * avg_length + 10 * good_hits - 8 * bad_hits
        
        if score > best_header_score:
            best_header_score = score
            best_header_idx = idx
            best_headers = headers
    
    return True, best_header_idx, best_headers


def _is_structured_data(rows: List[List[str]]) -> bool:
    """Check if the data looks like a structured table."""
    if len(rows) < 2:
        return False
    
    # Check if rows have consistent column counts
    col_counts = [len(row) for row in rows if any(c.strip() for c in row)]
    if col_counts:
        unique_counts = sorted(set(col_counts))
        # Be more tolerant: allow variations (e.g., continuation lines)
        if len(unique_counts) > 1 and (max(unique_counts) - min(unique_counts)) > 4:
            return False
    
    # Check if there's some pattern in the data (not just random text)
    non_empty_cells = sum(1 for row in rows for cell in row if cell.strip())
    if non_empty_cells < int(len(rows) * 1.2):  # Relaxed density requirement
        return False
    
    return True


def _detect_row_relationships(headers: List[str], rows: List[List[str]]) -> List[List[List[str]]]:
    """Automatically detect how rows are related and group them intelligently."""
    
    if not rows:
        return []
    
    # Analyze row patterns to understand relationships
    def analyze_row_pattern(row: List[str]) -> dict:
        """Analyze a row to understand its structure and content."""
        non_empty = [i for i, c in enumerate(row) if c.strip()]
        content_types = []
        
        for i, cell in enumerate(row):
            if not cell.strip():
                continue
            cell_lower = cell.lower()
            
            # Detect content types
            if any(char.isdigit() for char in cell):
                if any(char == '.' for char in cell):
                    content_types.append(('number', i))
                else:
                    content_types.append(('integer', i))
            elif len(cell) <= 20 and any(char.isupper() for char in cell):
                content_types.append(('code', i))
            elif any(phrase in cell_lower for phrase in ['origin', 'country', 'commodity', 'export', 'cpc']):
                content_types.append(('metadata', i))
            else:
                content_types.append(('text', i))
        
        return {
            'non_empty_count': len(non_empty),
            'content_types': content_types,
            'has_numbers': any(t[0] in ['number', 'integer'] for t in content_types),
            'has_codes': any(t[0] == 'code' for t in content_types),
            'has_metadata': any(t[0] == 'metadata' for t in content_types)
        }
    
    # Group rows by similarity
    row_groups = []
    current_group = []
    
    for row in rows:
        pattern = analyze_row_pattern(row)
        
        # Start new group if this row looks like a main data row
        if (pattern['has_numbers'] and pattern['non_empty_count'] >= 3) or \
           (pattern['has_codes'] and pattern['non_empty_count'] >= 2):
            
            if current_group:
                row_groups.append(current_group)
            current_group = [row]
        else:
            # This looks like continuation/supplementary data
            if current_group:
                current_group.append(row)
            else:
                # Start a new group if we don't have one yet
                current_group = [row]
    
    # Add the last group
    if current_group:
        row_groups.append(current_group)
    
    return row_groups


def _tables_to_dicts(tables: List[pd.DataFrame], page_title: str | None) -> List[Dict]:
    results: List[Dict] = []
    
    for df in tables:
        df = df.copy()
        
        # Detect if this is a structured table
        is_table, header_idx, headers = _detect_table_structure(df)
        
        if not is_table:
            # Fallback: attempt to coerce a simple table when Camelot returns grids
            df_nonempty = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            if df_nonempty.shape[0] >= 2 and df_nonempty.shape[1] >= 3:
                header_row_idx = 0
                headers = [str(c).strip() for c in df_nonempty.iloc[header_row_idx].tolist()]
                data_part = df_nonempty.iloc[header_row_idx + 1 :]
                data_rows = [["" if pd.isna(v) else str(v).strip() for v in row] for row in data_part.values.tolist()]
            else:
                continue
        
        # Extract data rows
        if header_idx >= 0:
            data_rows = []
            for i in range(header_idx + 1, len(df)):
                row = df.iloc[i]
                data_rows.append([str(c).strip() if pd.notna(c) else "" for c in row])
        else:
            # Fallback: use first row as header
            headers = [str(c).strip() for c in df.columns]
            data_rows = [["" if pd.isna(v) else str(v).strip() for v in row] for row in df.values.tolist()]
        
        # Check if this looks like structured data
        if not _is_structured_data(data_rows):
            continue
        
        # Normalize headers (remove duplicates, clean up)
        clean_headers = []
        seen = set()
        for h in headers:
            clean_h = h.strip()
            if clean_h and clean_h not in seen:
                clean_headers.append(clean_h)
                seen.add(clean_h)
        
        # Pad rows to match header count
        max_cols = len(clean_headers)
        padded_rows = []
        for row in data_rows:
            if len(row) < max_cols:
                row = row + [""] * (max_cols - len(row))
            elif len(row) > max_cols:
                row = row[:max_cols]
            padded_rows.append(row)
        
        # Filter out sentence-like narrative rows not part of table body
        def is_sentence_like(r: List[str]) -> bool:
            text = " ".join(c for c in r if c).strip()
            if not text:
                return False
            letters = sum(1 for ch in text if ch.isalpha())
            digits = sum(1 for ch in text if ch.isdigit())
            words = [w for w in text.split() if w]
            stopwords = {"the", "and", "of", "to", "by", "this", "are", "is", "from", "for", "with", "in"}
            stop_hits = sum(1 for w in words if w.lower() in stopwords)
            # Narrative if long, few digits, and many stopwords
            return (len(words) >= 6) and (digits <= 1) and (stop_hits >= 2)

        body_rows = [r for r in padded_rows if not is_sentence_like(r)]
        if not body_rows:
            body_rows = padded_rows

        # Group related rows
        grouped_rows = _detect_row_relationships(clean_headers, body_rows)
        
        # Only keep tables with meaningful data
        if grouped_rows and any(len(group) > 0 for group in grouped_rows):
            results.append({
                "title": page_title,
                "headers": clean_headers,
                "rows": grouped_rows
            })
    
    return results


def _extract_page_title(text: str) -> str | None:
    """Heuristically extract a human-friendly title from page text without hardcoding.

    Strategy (score-based, generic):
    - Prefer short lines (<= 80 chars) with many TitleCase words and few digits/symbols.
    - Penalize lines with email/URL patterns or contact info characteristics.
    - Avoid lines dominated by digits/punctuation.
    """
    if not text:
        return None
    candidates = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def score_line(ln: str) -> float:
        if len(ln) > 120:
            return -1e6
        letters = sum(1 for ch in ln if ch.isalpha())
        digits = sum(1 for ch in ln if ch.isdigit())
        if letters < 5:
            return -1e6
        tokens = [t for t in ln.replace("/", " ").replace("-", " ").split() if t]
        titlecase_tokens = sum(1 for t in tokens if len(t) >= 2 and t[0].isupper() and t[1:].islower())
        uppercase_tokens = sum(1 for t in tokens if t.isupper() and len(t) >= 3)
        colon_penalty = ln.count(":") * 1.5
        symbol_penalty = sum(ln.count(sym) for sym in ["@", "http", "www.", "+"]) * 3.0
        digit_penalty = max(0, digits - 2) * 0.5
        length_penalty = max(0, len(ln) - 80) * 0.2

        base = 2.0 * titlecase_tokens + 1.0 * uppercase_tokens + 0.3 * letters
        score = base - (colon_penalty + symbol_penalty + digit_penalty + length_penalty)
        return score

    best = None
    best_score = -1e9
    for ln in candidates:
        s = score_line(ln)
        if s > best_score:
            best_score = s
            best = ln
    return best if best_score > 0 else None


def _extract_page_header_block(text: str, first_table_headers: List[str]) -> List[str]:
    """Extract the leading header block (headline and key details) generically.

    Heuristics (no hardcoding of specific fields):
    - Take the first consecutive non-empty lines from the page top.
    - Stop when we likely reach the table area, detected by a line that contains
      multiple header tokens from the first detected table headers.
    - Limit lines to reasonable length to avoid full paragraphs.
    """
    if not text:
        return []

    # Prepare a simple token set from table headers for stop condition
    header_tokens = []
    for h in first_table_headers or []:
        parts = [p.strip() for p in str(h).split() if p.strip()]
        header_tokens.extend(parts)
    header_tokens = [t.lower() for t in header_tokens if t]

    lines = [ln.strip() for ln in text.splitlines()]
    header_lines: List[str] = []
    started = False
    for ln in lines:
        raw = ln.strip()
        if not raw:
            # Allow a single blank inside header once we've started, but stop on second
            if started:
                break
            else:
                continue

        # Skip very long lines that are likely paragraphs
        if len(raw) > 160:
            if started:
                break
            else:
                continue

        # Stop when we detect a line that is likely the table header line
        if header_tokens:
            match_count = sum(1 for tok in header_tokens if tok in raw.lower())
            # If many header tokens appear together, we likely reached the table
            if match_count >= max(3, int(0.5 * len(set(header_tokens)))):
                break

        header_lines.append(raw)
        started = True
        # Cap the number of lines to avoid capturing entire page
        if len(header_lines) >= 10:
            break

    # Trim trailing blank lines if any snuck in
    while header_lines and not header_lines[-1]:
        header_lines.pop()
    return header_lines


def _extract_title_above_table(text: str, first_table_headers: List[str]) -> str | None:
    """Find the line immediately above the table header in the page text.

    We detect the first line that strongly matches the table header tokens and
    then return the nearest non-empty line just above it. This avoids picking
    company contact info and aims for the section title like "Commercial Invoice".
    """
    if not text or not first_table_headers:
        return None

    lines = [ln.rstrip() for ln in text.splitlines()]
    # Prepare token set
    header_tokens: List[str] = []
    for h in first_table_headers:
        # keep multi-word headers as well as individual words
        h_str = str(h).strip()
        if h_str:
            header_tokens.append(h_str.lower())
            header_tokens.extend([p.strip().lower() for p in h_str.split() if p.strip()])

    header_tokens = [t for t in header_tokens if t and t not in {"", ":"}]
    if not header_tokens:
        return None

    # Map each token to earliest line index it appears in
    token_first_idx: List[int] = []
    for tok in header_tokens:
        idx = -1
        tok_lower = tok.lower()
        for i, ln in enumerate(lines):
            if tok_lower in ln.lower():
                idx = i
                break
        if idx >= 0:
            token_first_idx.append(idx)

    if not token_first_idx:
        return None

    # Heuristic table start: take the 25th percentile of first-occurrence indices
    token_first_idx.sort()
    q_index = max(0, int(0.25 * (len(token_first_idx) - 1)))
    header_idx = token_first_idx[q_index]

    # Walk upwards to find the closest non-empty, reasonably short line
    for j in range(header_idx - 1, -1, -1):
        candidate = lines[j].strip()
        if not candidate:
            continue
        if len(candidate) > 120:
            continue
        # Avoid lines with contact patterns
        low = candidate.lower()
        if any(p in low for p in ["@", "http", "www.", "+", "fax:", "tel:"]):
            continue
        return candidate
    return None


def _extract_invoice_metadata(text: str, first_table_headers: List[str]) -> Dict[str, str | None]:
    """Extract title, date, time, and invoice number from the page text.

    Heuristics:
    - Title: Prefer a line containing both 'commercial' and 'invoice'.
      Else fall back to the line above the table header.
    - Date/Time: Look for 'Dated:' and 'Time:' with nearby values.
    - Invoice No: Look for 'Invoice No' followed by a code.
    """
    if not text:
        return {"title": None, "date": None, "time": None, "invoice_no": None}

    title = None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        low = ln.lower()
        if "commercial" in low and "invoice" in low and len(ln) <= 80:
            title = ln
            break

    precise = _extract_title_above_table(text, first_table_headers)
    if not title:
        title = precise

    # Date and Time
    date = None
    time_val = None
    # Pattern variants like: Dated: 05 Jan 2024  Time: 08:19
    m = re.search(r"dated\s*:\s*([0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4}).*?time\s*:\s*([0-9]{1,2}:[0-9]{2})",
                  text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        date, time_val = m.group(1), m.group(2)
    else:
        md = re.search(r"dated\s*:\s*([0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4})", text, flags=re.IGNORECASE)
        mt = re.search(r"time\s*:\s*([0-9]{1,2}:[0-9]{2})", text, flags=re.IGNORECASE)
        if md:
            date = md.group(1)
        if mt:
            time_val = mt.group(1)

    # Invoice No
    invoice_no = None
    mi = re.search(r"invoice\s*no\s*[:#]?\s*([A-Za-z0-9\-\/]+)", text, flags=re.IGNORECASE)
    if mi:
        invoice_no = mi.group(1)

    return {"title": title, "date": date, "time": time_val, "invoice_no": invoice_no}

def extract_tables_digital(file_path: str, page_number: int) -> List[Dict]:
    """Extract tables from digital PDFs using Camelot."""
    all_tables: List[Dict] = []
    
    # Try to infer a title from the page text
    try:
        page_text = extract_page_text(file_path, page_number)
    except Exception:
        page_text = ""
    page_title = _extract_page_title(page_text)

    # Try different table extraction methods
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(file_path, pages=str(page_number), flavor=flavor)
            if tables and len(tables) > 0:
                dfs = [t.df for t in tables]
                extracted_tables = _tables_to_dicts(dfs, page_title)
                all_tables.extend(extracted_tables)
        except Exception:
            continue
    
    # Attach a header block using the line immediately above the table header
    if all_tables:
        first_headers = all_tables[0].get("headers", [])
        meta = _extract_invoice_metadata(page_text, first_headers)
        header_block_lines = _extract_page_header_block(page_text, first_headers)
        chosen_title = meta.get("title") or (header_block_lines[0] if header_block_lines else page_title)
        all_tables[0]["page_header"] = {
            "title": chosen_title,
            "date": meta.get("date"),
            "time": meta.get("time"),
            "invoice_no": meta.get("invoice_no"),
            "lines": header_block_lines or ([chosen_title] if chosen_title else []),
        }

        # Remove header-like lines that may have been captured inside the first table rows
        header_lines = all_tables[0].get("page_header", {}).get("lines", [])
        if header_lines and all_tables[0].get("rows"):
            header_blob_tokens = set()
            for hl in header_lines:
                for tok in str(hl).lower().replace("/", " ").replace("-", " ").split():
                    tok = tok.strip(' ,.;:|()')
                    if tok:
                        header_blob_tokens.add(tok)
            cleaned_groups: List[List[List[str]]] = []
            for gi, group in enumerate(all_tables[0]["rows"]):
                new_group: List[List[str]] = []
                for ri, row in enumerate(group):
                    non_empty = [c.strip() for c in row if str(c).strip()]
                    joined = " ".join(non_empty).strip()
                    if gi == 0 and ri <= 2 and joined:
                        # Drop row if it's equal to any header line
                        if any(joined == hl.strip() for hl in header_lines):
                            continue
                        # Drop if majority of tokens appear in header blob (split header text case)
                        row_tokens = [t.strip(' ,.;:|()').lower() for t in joined.split() if t.strip()]
                        if row_tokens:
                            match = sum(1 for t in row_tokens if t in header_blob_tokens)
                            if match >= max(3, int(0.6 * len(row_tokens))):
                                continue
                        # Also drop rows with a single non-empty cell matching the title
                        title_text = (all_tables[0].get("page_header") or {}).get("title") or ""
                        if len(non_empty) == 1 and non_empty[0] == title_text:
                            continue
                    new_group.append(row)
                if new_group:
                    cleaned_groups.append(new_group)
            if cleaned_groups:
                all_tables[0]["rows"] = cleaned_groups

        # Avoid duplicating the page title as a per-table title
        for t in all_tables:
            t["title"] = None

    return all_tables

