"""Split assistant responses into bounded windows, preserving every target token."""
from __future__ import annotations


def response_windows(tokenizer, example: dict, max_length: int) -> list[dict]:
    prompt = example["instruction"]
    if example.get("input"):
        prompt += "\n\n" + example["input"]
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("windowed SFT requires a tokenizer with a chat template")
    user = [{"role": "user", "content": prompt}]
    prefix = list(tokenizer.apply_chat_template(user, tokenize=True, add_generation_prompt=True))
    full = list(tokenizer.apply_chat_template(user + [{"role": "assistant", "content": example["output"]}],
                                               tokenize=True, add_generation_prompt=False))
    if full[:len(prefix)] != prefix:
        raise ValueError("chat template's prompt is not a prefix of the full conversation")
    target = full[len(prefix):]
    available = max_length - len(prefix)
    if available < 1:
        raise ValueError(f"prompt has {len(prefix)} tokens, leaving no room at max_length={max_length}")
    if not target:
        raise ValueError("assistant response has no tokens")
    result = []
    offset = 0
    while offset < len(target):
        # Repeat a short tail of prior reasoning as context, but never score it twice.
        context_count = min(offset, available // 4, 512)
        context = target[offset - context_count:offset]
        chunk = target[offset:offset + available - context_count]
        ids = prefix + context + chunk
        result.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                       "labels": [-100] * (len(prefix) + len(context)) + chunk})
        offset += len(chunk)
    assert sum(sum(label != -100 for label in window["labels"]) for window in result) == len(target)
    return result
