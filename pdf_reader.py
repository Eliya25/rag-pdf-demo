"""Bounded PDF extraction with page-aware text chunks."""
import io
import re

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PdfReader:
    def __init__(self, source="./pdfs/rag-pdf.pdf", max_file_mb: int = 10,
                 max_pages: int = 100, max_text_chars: int = 500_000):
        max_bytes = max_file_mb * 1024 * 1024
        if hasattr(source, "read"):
            # Read at most one byte beyond the limit so oversized streams do
            # not need to be fully materialized before rejection.
            data = source.read(max_bytes + 1)
        else:
            with open(source, "rb") as file:
                data = file.read(max_bytes + 1)
        if not data.startswith(b"%PDF-"):
            raise ValueError("The uploaded file is not a valid PDF.")
        if len(data) > max_bytes:
            raise ValueError(f"PDF exceeds the {max_file_mb} MB limit.")
        try:
            self.reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
            if self.reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")
            self.page_count = len(self.reader.pages)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("Could not read this PDF. It may be corrupted.") from error
        if self.page_count > max_pages:
            raise ValueError(f"PDF exceeds the {max_pages} page limit.")
        self.pages_text = ""
        self._pages = None
        self.max_text_chars = max_text_chars

    def _extract_pages(self):
        if self._pages is None:
            pages = []
            total_chars = 0
            try:
                for number, page in enumerate(self.reader.pages, 1):
                    raw_text = (page.extract_text() or "").replace("\r\n", "\n")
                    paragraphs = [re.sub(r"\s+", " ", part).strip()
                                  for part in re.split(r"\n\s*\n", raw_text)]
                    text = "\n\n".join(part for part in paragraphs if part)
                    total_chars += len(text)
                    if total_chars > self.max_text_chars:
                        raise ValueError("PDF contains too much extractable text.")
                    if text:
                        pages.append((number, text))
            except ValueError:
                raise
            except Exception as error:
                raise ValueError("PDF text extraction failed.") from error
            self._pages = pages
            self.pages_text = "\n".join(text for _, text in pages)
        return self._pages

    def extract_text(self):
        self._extract_pages()
        return self.pages_text

    def extract_small_portion_of_the_pdf(self, min=0, max=None):
        return self.extract_text()[min:max]

    def get_paragraphs(self):
        return [part.strip() for part in self.extract_text().split("\n") if part.strip()]

    def get_chunks(self, chunk_size=800, chunk_overlap=120, max_chunks=500):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Invalid chunk size or overlap")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", " ", ""], keep_separator="end",
        )
        chunks = []
        for page_number, text in self._extract_pages():
            for part in splitter.split_text(text):
                if part.strip():
                    chunks.append({"content": part.strip(), "page": page_number,
                                   "chunk": len(chunks) + 1})
                    if len(chunks) > max_chunks:
                        raise ValueError(f"PDF exceeds the {max_chunks} chunk limit.")
        return chunks
