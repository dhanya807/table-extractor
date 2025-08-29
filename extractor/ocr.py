from __future__ import annotations

import io
import os
import json
import re
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

from .pdf_utils import rasterize_page_image, extract_page_text
from .digital import _extract_invoice_metadata as _extract_invoice_metadata_digital


def _gemini_available() -> bool:
    """Check if Gemini API is available."""
    return bool(os.getenv("GEMINI_API_KEY"))


def _detect_table_structure_from_text(text: str) -> List[Dict]:
    """Intelligently detect table structure from OCR text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return []
    
    # Analyze line patterns to find potential table structure
    potential_tables = []
    current_table = {"headers": [], "rows": [], "title": None}
    
    for line in lines:
        # Check if line looks like a header (contains multiple meaningful words, not mostly numbers)
        words = line.split()
        if len(words) >= 3:
            numeric_count = sum(1 for word in words if word.replace('.', '').replace(',', '').replace('-', '').isdigit())
            if numeric_count < len(words) * 0.4:  # Less than 40% numbers
                # This could be a header row
                if current_table["headers"]:
                    # We already have headers, this might be a new table
                    if current_table["rows"]:
                        potential_tables.append(current_table)
                    current_table = {"headers": [], "rows": [], "title": None}
                current_table["headers"] = words
            else:
                # This looks like data
                if current_table["headers"]:
                    current_table["rows"].append(words)
    
    # Add the last table
    if current_table["headers"] and current_table["rows"]:
        potential_tables.append(current_table)
    
    # Convert to standard format
    result = []
    for table in potential_tables:
        if len(table["rows"]) > 0:
            result.append({
                "title": table.get("title"),
                "headers": table["headers"],
                "rows": [table["rows"]]  # Single group for simple tables
            })
    
    return result


def _call_gemini_table_ocr(png_bytes: bytes) -> Optional[List[Dict]]:
    """Use Gemini AI to extract tables from image with generic approach."""
    try:
        import google.generativeai as genai
    except Exception:
        return None
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        image = Image.open(io.BytesIO(png_bytes))
        
        # Generic prompt that works for any document type
        prompt = """
        Analyze this image and extract any structured table data you can find.
        
        Look for:
        - Tables with headers and rows
        - Lists with consistent structure
        - Any organized data in rows and columns
        
        Return ONLY a JSON object with this exact structure:
        {
          "tables": [
            {
              "title": "table title or null",
              "headers": ["header1", "header2", ...],
              "rows": [
                [
                  ["row1cell1", "row1cell2", ...],
                  ["continuation1", "continuation2", ...]
                ],
                [
                  ["row2cell1", "row2cell2", ...],
                  ["continuation1", "continuation2", ...]
                ]
              ]
            }
          ]
        }
        
        Each row group should contain the main row and any continuation rows that belong to it.
        If no tables found, return {"tables": []}.
        Focus on finding any structured data, regardless of document type.
        """
        
        response = model.generate_content([prompt, image])
        text = response.text or ""
        
        # Attempt to locate JSON in the response
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        
        obj = json.loads(match.group(0))
        tables = obj.get("tables")
        
        if isinstance(tables, list):
            normalized: List[Dict] = []
            for t in tables:
                if not isinstance(t, dict):
                    continue
                    
                title = t.get("title")
                headers = t.get("headers", [])
                rows = t.get("rows", [])
                
                if not isinstance(headers, list) or not isinstance(rows, list):
                    continue
                
                # Normalize headers and rows
                headers = [str(h).strip() for h in headers if h]
                rows = [[str(v).strip() for v in r if v] for r in rows if isinstance(r, list)]
                
                # Only include tables with meaningful data
                if headers and any(len(row) > 0 for row in rows):
                    normalized.append({
                        "title": title,
                        "headers": headers,
                        "rows": rows
                    })
            
            return normalized
            
    except Exception:
        return None
    
    return None


def _tesseract_fallback(png_bytes: bytes) -> List[Dict]:
    """Fallback OCR using Tesseract TSV to reconstruct table rows/columns.

    Generic approach:
    - Use image_to_data to get word boxes.
    - Cluster words into rows by y position.
    - Within each row, split into cells by large x gaps.
    - Choose header as the first row with >=3 cells and not mostly numeric.
    - Group continuation lines that have <50% of header cell count into the previous row.
    """
    try:
        import pytesseract
    except Exception:
        return []

    from PIL import Image
    import io
    import pandas as pd

    try:
        image = Image.open(io.BytesIO(png_bytes))
        tsv = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        if tsv is None or len(tsv) == 0:
            return []

        # Keep confident words
        df = tsv[(tsv.conf.fillna(-1) > 30) & tsv.text.notna() & (tsv.text.str.strip() != "")].copy()
        if df.empty:
            return []

        # Compute row clusters by top value
        df["cy"] = df["top"] + df["height"] / 2.0
        df = df.sort_values(["cy", "left"]).reset_index(drop=True)

        rows: List[List[Dict]] = []
        current: List[Dict] = []
        prev_cy = None
        row_tol = max(5, int(df["height"].median() * 0.6))
        for _, w in df.iterrows():
            cy = int(w["cy"]) if pd.notna(w["cy"]) else 0
            item = {"x": int(w["left"]), "y": int(w["top"]), "text": str(w["text"]).strip()}
            if prev_cy is None or abs(cy - prev_cy) <= row_tol:
                current.append(item)
                prev_cy = cy if prev_cy is None else int((prev_cy + cy) / 2)
            else:
                if current:
                    rows.append(sorted(current, key=lambda d: d["x"]))
                current = [item]
                prev_cy = cy
        if current:
            rows.append(sorted(current, key=lambda d: d["x"]))

        # Convert row word lists into cell arrays by detecting big x-gaps
        def words_to_cells(words: List[Dict]) -> List[str]:
            if not words:
                return []
            xs = [w["x"] for w in words]
            gaps = [xs[i+1]-xs[i] for i in range(len(xs)-1)]
            if gaps:
                gap_threshold = max(20, int(pd.Series(gaps).quantile(0.8)))
            else:
                gap_threshold = 40
            cells: List[str] = []
            buf: List[str] = []
            for i, w in enumerate(words):
                buf.append(w["text"])
                if i == len(words)-1 or (words[i+1]["x"] - w["x"]) > gap_threshold:
                    cells.append(" ".join(buf).strip())
                    buf = []
            return cells

        cell_rows: List[List[str]] = [words_to_cells(r) for r in rows]
        # Find header row with keyword preference
        header_idx = -1
        for i, r in enumerate(cell_rows):
            if len(r) >= 3:
                numeric_count = sum(1 for c in r if c.replace('.', '').replace(',', '').replace('-', '').isdigit())
                if numeric_count < len(r) * 0.5:
                    cand = " ".join(s.lower() for s in r)
                    good_kw = {"carton","order","order no","part","part number","description","unit","unit price","qty","quantity","sales","value","customs"}
                    bad_kw = {"dated","time","invoice","application","information"}
                    score = sum(1 for kw in good_kw if kw in cand) - sum(1 for kw in bad_kw if kw in cand)
                    if score >= 1:
                        header_idx = i
                        break
        if header_idx < 0:
            return [{"title": None, "headers": [], "rows": [cell_rows]}]

        headers = [c.strip() for c in cell_rows[header_idx]]
        # Normalize header names to common invoice schema when possible
        normalized_headers = []
        for h in headers:
            hl = h.lower()
            if "carton" in hl:
                normalized_headers.append("Carton")
            elif "order" in hl:
                normalized_headers.append("Order No")
            elif ("part" in hl and "number" in hl) or hl in {"part", "pn", "p/n"}:
                normalized_headers.append("Part Number")
            elif "desc" in hl:
                normalized_headers.append("Description")
            elif ("unit" in hl and "price" in hl) or hl == "price":
                normalized_headers.append("Unit Price")
            elif hl in {"qty","quantity"}:
                normalized_headers.append("Qty")
            elif "sales" in hl and "value" in hl:
                normalized_headers.append("Sales Value")
            elif "customs" in hl and "value" in hl:
                normalized_headers.append("Customs Value")
            else:
                normalized_headers.append(h)
        headers = normalized_headers
        data_part = cell_rows[header_idx+1:]

        grouped: List[List[List[str]]] = []
        current_group: List[List[str]] = []
        for r in data_part:
            if len(r) >= max(2, int(0.5 * len(headers))):
                if current_group:
                    grouped.append(current_group)
                current_group = [r]
            else:
                if current_group:
                    current_group.append(r)
                else:
                    current_group = [r]
        if current_group:
            grouped.append(current_group)

        return [{"title": None, "headers": headers, "rows": grouped}]

    except Exception as e:
        print("OCR fallback error:", e)
        return []



def extract_tables_ocr(file_path: str, page_number: int) -> List[Dict]:
    """Extract tables from scanned PDFs using OCR with generic approach."""
    try:
        png_bytes, _ = rasterize_page_image(file_path, page_number, dpi=240)
    except Exception:
        return []
    
    # Try Gemini AI first if available
    if _gemini_available():
        tables = _call_gemini_table_ocr(png_bytes)
        if tables:
            return tables
    
    # Fallback to Tesseract
    tables = _tesseract_fallback(png_bytes)

    # Attach page header metadata similar to digital
    if tables:
        try:
            page_text = extract_page_text(file_path, page_number)
        except Exception:
            page_text = ""
        first_headers = tables[0].get("headers", []) if isinstance(tables[0], dict) else []
        meta = _extract_invoice_metadata_digital(page_text, first_headers)
        title = meta.get("title")
        if isinstance(tables[0], dict):
            tables[0]["page_header"] = {
                "title": title,
                "date": meta.get("date"),
                "time": meta.get("time"),
                "invoice_no": meta.get("invoice_no"),
                "lines": [title] if title else [],
            }
    return tables

