from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from config import settings

DEFAULT_CHUNK_SIZE_CHARS = max(200, settings.chunk_size_chars)
DEFAULT_CHUNK_OVERLAP_CHARS = max(0, settings.chunk_overlap_chars)


@dataclass
class TextChunk:
    index: int
    text: str
    char_start: int
    char_end: int


def iter_text_chunks(
    text: str,
    chunk_size_chars: int | None = None,
    overlap_chars: int | None = None,
) -> Iterable[TextChunk]:
    """Yield stable text chunks with character offsets into the original text."""
    raw = str(text or "")
    if not raw.strip():
        return []

    chunk_size = int(chunk_size_chars or DEFAULT_CHUNK_SIZE_CHARS)
    overlap = int(overlap_chars or DEFAULT_CHUNK_OVERLAP_CHARS)
    if chunk_size <= 0:
        return []
    overlap = max(0, min(overlap, max(0, chunk_size - 1)))

    n = len(raw)
    start = 0
    index = 0
    min_break_window = max(60, chunk_size // 2)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            # Try to end on a whitespace boundary to avoid mid-word cuts.
            window_start = min(start + min_break_window, end)
            whitespace = raw.rfind(" ", window_start, end)
            if whitespace > start + 20:
                end = whitespace

        if end <= start:
            end = min(start + chunk_size, n)
            if end <= start:
                break

        slice_text = raw[start:end]
        leading = len(slice_text) - len(slice_text.lstrip())
        trailing = len(slice_text) - len(slice_text.rstrip())
        chunk_start = start + leading
        chunk_end = end - trailing

        if chunk_end > chunk_start:
            chunk_text = raw[chunk_start:chunk_end]
            if chunk_text:
                yield TextChunk(index=index, text=chunk_text, char_start=chunk_start, char_end=chunk_end)
                index += 1

        if end >= n:
            break

        next_start = max(end - overlap, start + 1)
        start = next_start


def build_text_chunks(
    text: str,
    chunk_size_chars: int | None = None,
    overlap_chars: int | None = None,
) -> List[TextChunk]:
    return list(iter_text_chunks(text, chunk_size_chars, overlap_chars))
