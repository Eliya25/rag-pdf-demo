import math
import pytest

from vector_index import VectorIndex


def test_cosine_order_and_dimensions():
    index = VectorIndex()
    index.add_vector([1, 0], {"content": "east"})
    index.add_vector([0, 1], {"content": "north"})
    assert [doc["content"] for doc, _ in index.search([1, 0], 5)] == ["east", "north"]
    assert index.search([1, 0])[0][1] == pytest.approx(0)
    with pytest.raises(ValueError):
        index.add_vector([1], {"content": "bad"})
    with pytest.raises(ValueError):
        index.search([1])


def test_euclidean_and_validation():
    index = VectorIndex("euclidean")
    index.add_vector([3, 4], {"content": "far"})
    index.add_vector([0, 0], {"content": "near"})
    assert index.search([0, 0], 2)[0][0]["content"] == "near"
    assert index.search([0, 0], 2)[1][1] == pytest.approx(5)
    with pytest.raises(ValueError):
        index.add_vector([math.nan, 1], {"content": "bad"})


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k_even_when_empty(k):
    with pytest.raises(ValueError):
        VectorIndex().search([1], k)


def test_zero_cosine_vector_and_empty_result():
    index = VectorIndex()
    assert index.search([1]) == []
    with pytest.raises(ValueError):
        index.add_vector([0, 0], {"content": "bad"})
