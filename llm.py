"""The small open-weight model both articles run: Qwen2.5-0.5B-Instruct, on the CPU.

Pinned to one revision, so every run loads the same weights. The first run downloads about
1 GB from the Hugging Face Hub, anonymously; later runs load it from the local cache.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"  # batches line up at the end, where generation happens
    return tokenizer


def load_model(dtype: torch.dtype = torch.float32):
    return AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype).eval()


def chat(tokenizer, prompts: list[str]):
    """Token ids for each prompt as a one-turn chat, padded into one batch."""
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    return tokenizer.apply_chat_template(conversations, add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True, padding=True)


@torch.no_grad()
def greedy(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[list[int]]:
    """Always the most likely next token, for every prompt in one batch. Returns the new ids."""
    batch = chat(tokenizer, prompts)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                         temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
    return [strip(row[batch["input_ids"].shape[1]:].tolist(), tokenizer) for row in out]


def strip(ids: list[int], tokenizer) -> list[int]:
    """Cut a generated sequence at its first end-of-turn token; drop padding."""
    stops = {tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    for i, t in enumerate(ids):
        if t in stops:
            return ids[:i]
    return ids
