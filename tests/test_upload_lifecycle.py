import pytest

from upload_lifecycle import read_upload_bytes, should_index_upload


def test_same_upload_is_retried_after_failed_indexing():
    selected_id = "same-pdf-hash"
    # First attempt stored the selected hash before processing, but failed
    # before creating an index. The next rerun must try the identical upload.
    assert should_index_upload(selected_id, selected_id, None) is True
    successful_index = object()
    assert should_index_upload(selected_id, selected_id, successful_index) is False
    assert should_index_upload("new-pdf-hash", selected_id, successful_index) is True


def test_oversized_reported_upload_is_rejected_before_getvalue():
    class Upload:
        size = 11

        def getvalue(self):
            raise AssertionError("oversized upload must not be copied")

    with pytest.raises(ValueError, match="upload size limit"):
        read_upload_bytes(Upload(), max_bytes=10)


def test_upload_bytes_are_checked_against_actual_length():
    class Upload:
        size = 5

        def getvalue(self):
            return b"x" * 11

    with pytest.raises(ValueError, match="upload size limit"):
        read_upload_bytes(Upload(), max_bytes=10)
