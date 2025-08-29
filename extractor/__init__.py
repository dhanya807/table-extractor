"""Extractor package for PDF table extraction."""

from .pdf_utils import DocumentInfo, PageRange, load_document_info, iter_page_numbers
from .digital import extract_tables_digital
from .ocr import extract_tables_ocr
from .merger import merge_multi_page_tables, normalize_table

__all__ = [
    "DocumentInfo",
    "PageRange",
    "load_document_info",
    "iter_page_numbers",
    "extract_tables_digital",
    "extract_tables_ocr",
    "merge_multi_page_tables",
    "normalize_table",
]

