from __future__ import annotations

import json
import os
from typing import List, Optional

import click
from joblib import Parallel, delayed

from extractor import (
    DocumentInfo,
    PageRange,
    load_document_info,
    iter_page_numbers,
    extract_tables_digital,
    extract_tables_ocr,
    merge_multi_page_tables,
)

from formatters import to_json_file, to_markdown_file, to_html_file
from formatters.json_fmt import to_json_structure


def _parse_pages(pages: Optional[str]) -> PageRange:
    if not pages:
        return PageRange(None)
    # Format: "2,3" or "1-3,5"
    result: List[int] = []
    parts = [p.strip() for p in pages.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                continue
            if start <= end:
                result.extend(list(range(start, end + 1)))
        else:
            try:
                result.append(int(part))
            except ValueError:
                continue
    result = sorted(set(result))
    return PageRange(result or None)


def _extract_single_file(file_path: str, page_range: PageRange, output_formats: List[str], output_dir: str,
                         use_ocr: bool) -> List[str]:
    written: List[str] = []
    if not os.path.exists(file_path):
        click.echo(f"[WARN] File not found: {file_path}")
        return written
    try:
        info: DocumentInfo = load_document_info(file_path)
    except Exception as e:
        click.echo(f"[WARN] Failed to open {file_path}: {e}")
        return written

    per_page_tables = []
    for p in iter_page_numbers(page_range, info.page_count):
        try:
            tables = extract_tables_digital(file_path, p)
            if use_ocr and not tables:
                tables = extract_tables_ocr(file_path, p)
        except Exception as e:
            click.echo(f"[WARN] Extraction error {file_path} page {p}: {e}")
            tables = []
        per_page_tables.append(tables)

        base = os.path.splitext(os.path.basename(file_path))[0]
        out_base = os.path.join(output_dir, base)
        os.makedirs(output_dir, exist_ok=True)

        # Write per-page outputs
        if tables:
            data = to_json_structure(file_path, p, tables)
            if "json" in output_formats:
                out_json = f"{out_base}_page{p}.json"
                to_json_file(out_json, data)
                written.append(out_json)
            if "md" in output_formats:
                out_md = f"{out_base}_page{p}.md"
                to_markdown_file(out_md, file_path, p, tables)
                written.append(out_md)
            if "html" in output_formats:
                out_html = f"{out_base}_page{p}.html"
                to_html_file(out_html, file_path, p, tables)
                written.append(out_html)

    # Merge across pages and write combined outputs
    merged = merge_multi_page_tables(per_page_tables)

    # Only write merged file if any table was actually merged (i.e., the number of merged tables is less than the total tables)
    total_tables = sum(len(page_tables) for page_tables in per_page_tables)
    if merged and len(merged) < total_tables:
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_base = os.path.join(output_dir, base)
        data = {"metadata": {"file": os.path.basename(file_path), "page": None}, "tables": merged}
        if "json" in output_formats:
            out_json = f"{out_base}_merged.json"
            to_json_file(out_json, data)
            written.append(out_json)
        if "md" in output_formats:
            out_md = f"{out_base}_merged.md"
            to_markdown_file(out_md, file_path, 0, merged)
            written.append(out_md)
        if "html" in output_formats:
            out_html = f"{out_base}_merged.html"
            to_html_file(out_html, file_path, 0, merged)
            written.append(out_html)

    return written


@click.command()
@click.option('--files', multiple=True, required=True, type=click.Path(exists=False, dir_okay=False), help='Input PDF files')
@click.option('--pages', default=None, help='Comma/range list e.g., "2,3" or "1-3,5"')
@click.option('--output-format', default='json,md', help='Comma list: json,md,html')
@click.option('--output-dir', default='outputs', help='Directory for outputs')
@click.option('--ocr/--no-ocr', default=True, help='Enable OCR fallback for scanned PDFs')
@click.option('--jobs', default=-1, help='Parallel jobs for multi-file processing')
def main(files: List[str], pages: Optional[str], output_format: str, output_dir: str, ocr: bool, jobs: int):
    page_range = _parse_pages(pages)
    formats = [f.strip().lower() for f in output_format.split(',') if f.strip()]
    tasks = [(f, page_range, formats, output_dir, ocr) for f in files]
    results = Parallel(n_jobs=jobs)(delayed(_extract_single_file)(*t) for t in tasks)
    total_written = sum(len(r) for r in results)
    click.echo(f"Done. Wrote {total_written} files to {output_dir}.")


if __name__ == '__main__':
    main()