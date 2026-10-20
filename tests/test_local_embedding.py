import pytest
import torch
from threading import Lock

from local_embedding import LocalEmbedding
from vector_index import VectorIndex


def test_mean_pool_masks_padding_and_get_embeddings_normalizes_batches():
    embedding = LocalEmbedding.__new__(LocalEmbedding)
    embedding.device = "cpu"
    embedding._inference_lock = Lock()

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["one", "two"]
            assert kwargs["padding"] is True
            return {
                "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            }

    class Model:
        def __call__(self, **encoded):
            return type("Output", (), {"last_hidden_state": torch.tensor([
                [[3.0, 0.0], [0.0, 4.0], [1000.0, 1000.0]],
                [[0.0, 5.0], [1000.0, 1000.0], [1000.0, 1000.0]],
            ])})()

    embedding.tokenizer = Tokenizer()
    embedding.model = Model()
    output = embedding.get_embeddings(["one", "two"])
    assert output.shape == (2, 2)
    assert output.dtype == torch.float32
    assert torch.linalg.vector_norm(output, dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert output[0].tolist() == pytest.approx([0.6, 0.8])
    assert output[1].tolist() == pytest.approx([0.0, 1.0])


def test_build_index_batches_chunks_and_rejects_invalid_batch_size():
    embedding = LocalEmbedding.__new__(LocalEmbedding)
    embedding.distance_metric = "cosine"
    embedding.store = VectorIndex()
    calls = []

    def get_embeddings(texts):
        calls.append(list(texts))
        return torch.tensor([[1.0, 0.0] if text == "first" else [0.0, 1.0]
                             for text in texts])

    embedding.get_embeddings = get_embeddings
    chunks = [{"content": "first", "page": 1, "chunk": 1},
              {"content": "second", "page": 2, "chunk": 2}]
    embedding.build_index(chunks, batch_size=1)
    assert calls == [["first"], ["second"]]
    assert len(embedding.store) == 2
    assert embedding.search([0.0, 1.0], k=1)[0][0]["page"] == 2

    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            embedding.build_index(chunks, batch_size=invalid)
