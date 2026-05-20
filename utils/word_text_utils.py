from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


def _clean_text(value) -> str:
    return " ".join(str(value or "").strip().split())


def _iter_block_items(doc: DocumentObject):
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _table_to_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [_clean_text(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    formatted_rows = [" | ".join(cell if cell else "-" for cell in row) for row in rows]
    return "\n".join(formatted_rows)


def extract_word_text(docx_path):
    doc = Document(docx_path)
    sections = []

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = _clean_text(block.text)
            if text:
                sections.append(text)
        elif isinstance(block, Table):
            table_text = _table_to_text(block)
            if table_text:
                sections.append(table_text)

    return "\n\n".join(sections).strip()
