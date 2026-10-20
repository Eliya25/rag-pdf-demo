"""Small decisions for the Streamlit upload/index lifecycle."""


def should_index_upload(file_id: str, current_pdf_id: str | None, index) -> bool:
    """Index a new PDF, or retry the selected PDF when no index was built."""
    return file_id != current_pdf_id or index is None


def read_upload_bytes(upload, max_bytes: int) -> bytes:
    """Reject using Streamlit's reported size before copying uploaded bytes."""
    if upload.size > max_bytes:
        raise ValueError("PDF exceeds the configured upload size limit.")
    data = upload.getvalue()
    if len(data) > max_bytes:
        raise ValueError("PDF exceeds the configured upload size limit.")
    return data
