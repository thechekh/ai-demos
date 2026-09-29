"""Perplexity: how surprised a model is by your text.

Article: https://chekh.dev/writing/perplexity-how-surprised-a-model-is-by-your-text/
Run:     uv run python perplexity.py                (about ten minutes on a CPU)
         uv run python perplexity.py --charts-only  (redraw from results/)
         uv run pytest tests/test_perplexity.py

Qwen2.5-0.5B-Instruct's surprise, token by token, at true and false sentences; its
perplexity on news, the same news shuffled, Python code, random letters, its own writing
and a journalist's; the same 200 news sentences in five languages, where per-token
perplexity and total surprise disagree; and what the sentences before a sentence do to it.
"""

import inspect
import json
import math
import random
import sys
import textwrap
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import ACCENT, DANGER, INK, INK_2, MUTED, machine, save
from llm import chat, greedy, load_model, load_tokenizer
from tokens import DATA, NTREX, ntrex

SLUG = "perplexity-how-surprised-a-model-is-by-your-text"
START = "<|endoftext|>"  # a document boundary, so that the first token of a text is scored too
N_SENTENCES = 200


def bits_from_logits(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Surprise in bits: minus log2 of the probability of each actual token."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(-1, ids[:, None]).squeeze(-1) / math.log(2)


@torch.no_grad()
def surprises(model, prefix: list[int], ids: list[int]) -> list[float]:
    """The model's surprise at each token of `ids`, having read `prefix` first."""
    inputs = torch.tensor([prefix + ids])
    logits = model(inputs).logits[0, len(prefix) - 1:-1]  # each position predicts the next token
    return bits_from_logits(logits, torch.tensor(ids)).tolist()


def perplexity(bits: list[float]) -> float:
    """2 to the power of the average surprise: effective choices per token."""
    return 2 ** (sum(bits) / len(bits))


def text_bits(model, tokenizer, text: str, context: str = "") -> list[float]:
    prefix = tokenizer.encode(START + context, add_special_tokens=False)
    return surprises(model, prefix, tokenizer.encode(text, add_special_tokens=False))


def reply_bits(model, tokenizer, prompt: str, reply: str) -> list[float]:
    """Surprise at a reply to a chat prompt, whoever wrote the reply."""
    prefix = chat(tokenizer, [prompt])["input_ids"][0].tolist()
    return surprises(model, prefix, tokenizer.encode(reply, add_special_tokens=False))


# --- Experiments ----------------------------------------------------------------------------

TRUE_FALSE = [("The Eiffel Tower is in", " Paris", " Rome"),
              ("The capital of Australia is", " Canberra", " Sydney")]


def token_by_token(model, tokenizer) -> list[dict]:
    """Surprise at every token of a true and a false ending; the answer's bits are summed."""
    out = []
    for prefix, *answers in TRUE_FALSE:
        n_prefix = len(tokenizer.encode(prefix, add_special_tokens=False))
        for answer in answers:
            text = prefix + answer + "."
            ids = tokenizer.encode(text, add_special_tokens=False)
            bits = text_bits(model, tokenizer, text)
            answer_bits = sum(bits[n_prefix:-1])
            out.append({"text": text, "tokens": [tokenizer.decode([i]) for i in ids], "bits": bits,
                        "answer": answer.strip(), "answer tokens": list(range(n_prefix, len(ids) - 1)),
                        "answer bits": answer_bits, "answer probability": 2 ** -answer_bits})
            print(f"  {text:40s} the answer costs {answer_bits:5.1f} bits (1 in {2 ** answer_bits:,.0f})")
    return out


def shuffled(sentence: str, rng: random.Random) -> str:
    words = sentence.split()
    rng.shuffle(words)
    return " ".join(words)


def document_ids() -> list[str]:
    path = DATA / "DOCUMENT_IDS.tsv"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NTREX.rsplit("/", 1)[0] + "/DOCUMENT_IDS.tsv", path)
    return path.read_text(encoding="utf-8").splitlines()


def text_types(model, tokenizer) -> dict:
    rng = random.Random(0)
    news = ntrex("English")[:N_SENTENCES]
    out = {}

    def measure(name, texts):
        bits = [b for t in texts for b in text_bits(model, tokenizer, t)]
        out[name] = {"texts": len(texts), "tokens": len(bits), "perplexity": perplexity(bits)}
        print(f"  {name:36s} {len(bits):6,} tokens  perplexity {out[name]['perplexity']:8.1f}")

    measure("news sentences", news)
    measure("the same sentences, words shuffled", [shuffled(s, rng) for s in news])
    code = inspect.getsource(textwrap)
    measure("Python code (textwrap.py)", [code[i:i + 1500] for i in range(0, len(code), 1500)])
    letters = "abcdefghijklmnopqrstuvwxyz"
    measure("random letters", ["".join(rng.choice(letters + " ") for _ in range(120)) for _ in range(40)])
    measure("one word, repeated", [" ".join(["the"] * 60)])

    # the next sentence of a real news story: a journalist's, and the model's own, given the same start
    ids, lines = document_ids(), ntrex("English")
    starts = [i for i in range(1, len(lines)) if ids[i] == ids[i - 1] and (i == 1 or ids[i - 1] != ids[i - 2])][:40]
    prompts = [f"Continue this news report with its next sentence: {lines[i - 1]}" for i in starts]
    journalist = [lines[i] for i in starts]
    model_says = [tokenizer.decode(r) for r in greedy(model, tokenizer, prompts, max_new_tokens=40)]
    for name, replies in [("the next sentence, written by the journalist", journalist),
                          ("the next sentence, written by the model", model_says)]:
        bits = [b for p, r in zip(prompts, replies) for b in reply_bits(model, tokenizer, p, r)]
        out[name] = {"texts": len(replies), "tokens": len(bits), "perplexity": perplexity(bits), "example": replies[0]}
        print(f"  {name:36s} {len(bits):6,} tokens  perplexity {out[name]['perplexity']:8.1f}")
    out["example prompt"] = prompts[0]
    return out


def languages(model, tokenizer) -> dict:
    out = {}
    for language in ("English", "German", "Russian", "Chinese", "Hindi"):
        sentences = ntrex(language)[:N_SENTENCES]
        bits = [b for s in sentences for b in text_bits(model, tokenizer, s)]
        out[language] = {"tokens": len(bits), "perplexity per token": perplexity(bits), "total bits": sum(bits),
                         "bits per sentence": sum(bits) / len(sentences)}
        print(f"  {language:8s} {len(bits):6,} tokens  per-token perplexity {perplexity(bits):6.1f}  "
              f"total {sum(bits) / 8 / 1024:6.1f} KiB")
    return out


def context(model, tokenizer, documents: int = 40) -> dict:
    """Each sentence alone, then after the sentences before it in the same news story."""
    ids, lines = document_ids(), ntrex("English")
    alone, with_context = [], []
    doc_order = list(dict.fromkeys(ids))[:documents]
    for doc in doc_order:
        sentences = [line for line, d in zip(lines, ids) if d == doc]
        for k in range(1, min(len(sentences), 8)):
            alone += text_bits(model, tokenizer, sentences[k])
            with_context += text_bits(model, tokenizer, " " + sentences[k], context=" ".join(sentences[:k]))
    out = {"documents": len(doc_order), "tokens alone": len(alone), "tokens in context": len(with_context),
           "perplexity alone": perplexity(alone), "perplexity in context": perplexity(with_context)}
    print(f"  alone {out['perplexity alone']:.1f}, after the story so far {out['perplexity in context']:.1f}")
    return out


def compute() -> dict:
    tokenizer, model = load_tokenizer(), load_model()
    results: dict = {"machine": machine(), "torch": torch.__version__, "sentences": N_SENTENCES}
    results["token by token"] = token_by_token(model, tokenizer)
    results["text types"] = text_types(model, tokenizer)
    results["languages"] = languages(model, tokenizer)
    results["context"] = context(model, tokenizer)
    return results


# --- Charts ---------------------------------------------------------------------------------


def draw_surprise(results: dict) -> None:
    rows = results["token by token"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.8, 1.35 * len(rows) + 0.4))
    top = max(max(r["bits"]) for r in rows)
    for ax, r in zip(axes, rows):
        n, answer = len(r["tokens"]), set(r["answer tokens"])
        true_one = r is rows[0] or r is rows[2]
        colors = [(ACCENT if true_one else DANGER) if i in answer else MUTED for i in range(n)]
        ax.bar(range(n), r["bits"], color=colors, width=0.7)
        ax.set_xticks(range(n), [t.replace(" ", "") for t in r["tokens"]], fontsize=7.8)
        ax.set_ylim(0, top * 1.2)
        ax.set_yticks([0, 10, 20])
        ax.spines["left"].set_visible(False)
        last = max(answer)
        ax.text(last + 0.45, max(r["bits"][i] for i in answer) + top * 0.03,
                f"{r['answer']}: {r['answer bits']:.1f} bits", fontsize=7.8, color=INK, ha="left")
    axes[0].set_title("Surprise at each token, in bits (more bits = less expected)", fontsize=10)
    fig.tight_layout()
    save(fig, SLUG, "surprise")


def draw_types(results: dict) -> None:
    t = {k: v for k, v in results["text types"].items() if isinstance(v, dict)}
    names = sorted(t, key=lambda k: t[k]["perplexity"])
    values = [t[k]["perplexity"] for k in names]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = [ACCENT if "model" in k else (INK if "journalist" in k or k == "news sentences" else MUTED) for k in names]
    ax.barh(range(len(names)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(names)), names, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for i, v in enumerate(values):
        ax.text(v * 1.08, i, f"{v:,.1f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0.8, max(values) * 3)
    ax.set_xlabel("perplexity (log scale)")
    ax.set_title("Perplexity measures how predictable a text is, not how good", fontsize=10)
    save(fig, SLUG, "types")


def draw_languages(results: dict) -> None:
    lang = results["languages"]
    names = list(lang)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    per_token = [lang[n]["perplexity per token"] for n in names]
    per_sentence = [lang[n]["bits per sentence"] for n in names]
    for ax, values, title, fmt in [(axes[0], per_token, "perplexity per token", "{:.1f}"),
                                   (axes[1], per_sentence, "surprise per sentence (bits)", "{:.0f}")]:
        ax.barh(range(len(names)), values, color=[ACCENT if n == "English" else INK_2 for n in names], height=0.6)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0, max(values) * 1.25)
        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.02, i, fmt.format(v), va="center", fontsize=8)
    fig.suptitle(f"The same {results['sentences']} news sentences in five languages", fontsize=10)
    fig.tight_layout()
    save(fig, SLUG, "languages")


if __name__ == "__main__":
    path = Path(__file__).parent / "results" / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = compute()
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {path.name}")
    draw_surprise(results)
    draw_types(results)
    draw_languages(results)
