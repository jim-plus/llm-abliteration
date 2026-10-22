import unittest

import torch

from utils.clip import magnitude_clip


class MagnitudeClipTests(unittest.TestCase):
    def test_batch_matches_individual_vectors(self):
        vectors = torch.tensor([[1.0, -2.0, 100.0], [10.0, -20.0, 1000.0]])
        expected = torch.stack([magnitude_clip(row, 0.5) for row in vectors])

        torch.testing.assert_close(magnitude_clip(vectors, 0.5), expected)

    def test_prompt_is_independent_of_batch_companions(self):
        prompt = torch.tensor([1.0, -2.0, 100.0])
        companion = torch.tensor([10000.0, -20000.0, 1000000.0])

        torch.testing.assert_close(
            magnitude_clip(torch.stack([prompt, companion]), 0.5)[0],
            magnitude_clip(prompt, 0.5),
        )

    def test_single_vector_preserves_symmetric_clipping(self):
        vector = torch.tensor([-100.0, -2.0, 1.0, 3.0])

        torch.testing.assert_close(
            magnitude_clip(vector, 0.5), torch.tensor([-2.5, -2.0, 1.0, 2.5])
        )

    def test_preserves_dtype_shape_and_input(self):
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                vector = torch.tensor([[1.0, -2.0, 100.0]], dtype=dtype)
                original = vector.clone()
                result = magnitude_clip(vector, 0.5)

                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, vector.shape)
                self.assertEqual(result.device, vector.device)
                torch.testing.assert_close(vector, original)
                torch.testing.assert_close(
                    result, torch.tensor([[1.0, -2.0, 2.0]], dtype=dtype)
                )

    def test_percentile_one_leaves_each_vector_unchanged(self):
        vectors = torch.tensor([[1.0, -2.0, 100.0], [10.0, -20.0, 1000.0]])

        torch.testing.assert_close(magnitude_clip(vectors, 1.0), vectors)

    def test_percentile_zero_uses_each_vectors_minimum(self):
        vectors = torch.tensor([[1.0, -2.0, 100.0], [10.0, -20.0, 1000.0]])

        torch.testing.assert_close(
            magnitude_clip(vectors, 0.0),
            torch.tensor([[1.0, -1.0, 1.0], [10.0, -10.0, 10.0]]),
        )


if __name__ == "__main__":
    unittest.main()
