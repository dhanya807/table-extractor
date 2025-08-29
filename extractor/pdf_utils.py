from __future__ import annotations

import dataclasses
from typing import Iterator, Optional, Tuple

# Don't import fitz at module level to avoid import errors
# We'll import it when needed in the functions


@dataclasses.dataclass
class DocumentInfo:
    file_path: str
    page_count: int
    title: Optional[str]
    author: Optional[str]
    creator: Optional[str]
    producer: Optional[str]


@dataclasses.dataclass
class PageRange:
    pages: Optional[List[int]]  # 1-based page numbers; None means all pages


def _get_fitz():
    """Get the PyMuPDF fitz module with proper error handling."""
    try:
        import fitz
        return fitz
    except ImportError:
        try:
            import PyMuPDF as fitz
            return fitz
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")


def load_document_info(file_path: str) -> DocumentInfo:
    fitz = _get_fitz()
    doc = fitz.open(file_path)
    metadata = doc.metadata or {}
    info = DocumentInfo(
        file_path=file_path,
        page_count=doc.page_count,
        title=metadata.get("title"),
        author=metadata.get("author"),
        creator=metadata.get("creator"),
        producer=metadata.get("producer"),
    )
    doc.close()
    return info


def iter_page_numbers(page_range: PageRange, total_pages: int) -> Iterator[int]:
    if page_range.pages is None:
        for p in range(1, total_pages + 1):
            yield p
        return
    for p in page_range.pages:
        if 1 <= p <= total_pages:
            yield p


def extract_page_text(file_path: str, page_number: int) -> str:
    fitz = _get_fitz()
    doc = fitz.open(file_path)
    try:
        page = doc.load_page(page_number - 1)
        return page.get_text("text")
    finally:
        doc.close()


def rasterize_page_image(file_path: str, page_number: int, dpi: int = 200) -> Tuple[bytes, Tuple[int, int]]:
    fitz = _get_fitz()
    doc = fitz.open(file_path)
    try:
        page = doc.load_page(page_number - 1)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        return pix.tobytes("png"), (pix.width, pix.height)
    finally:
        doc.close()

