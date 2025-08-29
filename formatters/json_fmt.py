from __future__ import annotations

import json
import os
from typing import Dict, List


def to_json_structure(file_path: str, page_number: int, tables: List[Dict]) -> Dict:
    metadata: Dict = {
        "file": os.path.basename(file_path),
        "page": page_number,
    }

    # If extractors attached a page header, surface it in metadata and set table title
    if tables and isinstance(tables[0], dict):
        header = tables[0].get("page_header")
        if isinstance(header, dict):
            title = header.get("title") or (header.get("lines") or [None])[0]
            if title:
                metadata["title"] = title
                # Ensure first table's title is not null for downstream consumers
                tables[0]["title"] = title
            if header.get("date"):
                metadata["date"] = header.get("date")
            if header.get("time"):
                metadata["time"] = header.get("time")
            if header.get("invoice_no"):
                metadata["invoice_no"] = header.get("invoice_no")

    return {
        "metadata": metadata,
        "tables": tables,
    }


def to_json_file(output_path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

