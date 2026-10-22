import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from measure import welford_gpu_batched_multilayer_float32


class MeasurementClippingTests(unittest.TestCase):
    def test_clipped_measurements_are_independent_of_batch_size(self):
        # Build a tiny real model/tokenizer locally; no downloaded weights or GPU.
        backend = Tokenizer(
            WordLevel(
                {"<unk>": 0, "<eos>": 1, "one": 2, "two": 3, "three": 4, "four": 5},
                unk_token="<unk>",
            )
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<eos>",
        )
        config = LlamaConfig(
            vocab_size=6,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            eos_token_id=1,
            pad_token_id=1,
            bos_token_id=None,
            attn_implementation="eager",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = LlamaForCausalLM(config).eval()

        prompts = ["one two", "three four one two"]
        with torch.inference_mode():
            individual = welford_gpu_batched_multilayer_float32(
                prompts, "individual", model, tokenizer, [0, 1], batch_size=1, clip=0.5
            )
            batched = welford_gpu_batched_multilayer_float32(
                prompts, "batched", model, tokenizer, [0, 1], batch_size=2, clip=0.5
            )

        for layer in individual:
            with self.subTest(layer=layer):
                torch.testing.assert_close(
                    individual[layer], batched[layer], atol=1e-7, rtol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
