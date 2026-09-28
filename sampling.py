"""Why the same prompt gives different answers.

Article: https://chekh.dev/writing/why-the-same-prompt-gives-different-answers/
Run:     uv run python sampling.py                (about ten minutes on a 6-core CPU; each stage
                                                   is saved as it finishes, so a rerun resumes)
         uv run python sampling.py --charts-only  (redraw from results/)
         uv run pytest tests/test_sampling.py

A small open-weight model, Qwen2.5-0.5B-Instruct, run on the CPU: the probabilities it
gives the next token, what temperature and top-p do to them, how many different answers
fifty samples give at each temperature, and the same prompt answered greedily, alone and
inside a batch, in 32-bit and in 16-bit arithmetic.
"""

import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _common import ACCENT, DANGER, INK, INK_2, MUTED, machine, save
from llm import MODEL, REVISION, chat, greedy, load_model, load_tokenizer, strip

SLUG = "why-the-same-prompt-gives-different-answers"
RESULTS = Path(__file__).parent / "results"
PARTIAL = RESULTS / f"{SLUG}.partial.json"


# --- One step: from scores to a token -------------------------------------------------------


def probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax of the scores over the temperature: low sharpens, high flattens."""
    return torch.softmax(logits / temperature, dim=-1)


def next_token(logits: torch.Tensor, temperature: float, top_p: float,
               generator: torch.Generator) -> torch.Tensor:
    """Greedy at temperature 0; otherwise a sample from the top_p set."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = probabilities(logits, temperature)
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p  # keeps the top token
    kept = sorted_probs * keep
    kept = kept / kept.sum(dim=-1, keepdim=True)
    pick = torch.multinomial(kept, 1, generator=generator)
    return order.gather(-1, pick).squeeze(-1)


@torch.no_grad()
def sample(model, tokenizer, prompt: str, n: int, max_new_tokens: int, temperature: float, top_p: float = 1.0,
           seed: int = 0) -> list[str]:
    """n answers to one prompt, generated together, one token at a time."""
    generator = torch.Generator().manual_seed(seed)
    ids = chat(tokenizer, [prompt])["input_ids"].repeat(n, 1)
    out = model(input_ids=ids, use_cache=True)
    new = []
    for _ in range(max_new_tokens):
        token = next_token(out.logits[:, -1, :], temperature, top_p, generator)
        new.append(token)
        out = model(input_ids=token[:, None], past_key_values=out.past_key_values, use_cache=True)
    rows = torch.stack(new, dim=1).tolist()
    return [tokenizer.decode(strip(row, tokenizer)) for row in rows]


# --- Experiments ----------------------------------------------------------------------------

FIRST_TOKEN_PROMPTS = ["Name a fruit. Answer with one word.", "Pick a number from 1 to 9. Answer with the number only."]
TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0, 1.3]
SEA = "Write one sentence about the sea."
PROMPTS = [
    SEA, "Give me three tips for learning a language.", "What is a hash table? Answer briefly.",
    "Describe a sunset in two sentences.", "Suggest a name for a bakery and explain it.",
    "Why is the sky blue? One paragraph.", "Write a haiku about autumn.", "Explain recursion to a ten-year-old.",
    "What should I pack for a weekend hike?", "Summarise the plot of Romeo and Juliet in three sentences.",
    "Give two arguments for and against remote work.", "How does a refrigerator keep food cold?",
    "Write a short thank-you note to a teacher.", "What are the benefits of drinking water?",
    "Invent a new board game and describe its rules.", "Explain what an API is to a non-programmer.",
]
N_TOKENS = 80


@torch.no_grad()
def first_token(model, tokenizer, prompt: str, k: int = 10) -> dict:
    logits = model(**chat(tokenizer, [prompt])).logits[0, -1, :]
    top = probabilities(logits, 1.0).topk(k)
    ids = top.indices.tolist()
    out = {"tokens": [tokenizer.decode([i]) for i in ids],
           "at temperature": {str(t): probabilities(logits, t)[ids].tolist() for t in (0.3, 0.7, 1.0, 1.5)}}
    ranked = probabilities(logits, 1.0).sort(descending=True).values
    out["kept by top-p"] = {str(p): int(((ranked.cumsum(0) - ranked) < p).sum()) for p in (0.5, 0.8, 0.9, 0.95)}
    return out


@torch.no_grad()
def continuations(model, tokenizer, prompt: str, starts: list[str], k: int = 3) -> dict:
    """Force each first token, then list the likeliest second ones: how the word goes on."""
    base = chat(tokenizer, [prompt])["input_ids"]
    out = {}
    for start in starts:
        ids = torch.cat([base, torch.tensor([[tokenizer.convert_tokens_to_ids(start)]])], dim=1)
        top = probabilities(model(input_ids=ids).logits[0, -1, :], 1.0).topk(k)
        out[start] = [[tokenizer.decode([int(i)]), float(q)] for q, i in zip(top.values, top.indices)]
    return out


def distinct_answers(model, tokenizer, n: int = 50, max_new_tokens: int = 24) -> dict:
    out = {"prompt": SEA, "n": n, "max_new_tokens": max_new_tokens, "temperatures": TEMPERATURES, "runs": {}}
    greedy_answer = sample(model, tokenizer, SEA, 1, max_new_tokens, 0.0)[0]
    for top_p in (1.0, 0.8):
        for t in TEMPERATURES:
            answers = sample(model, tokenizer, SEA, n, max_new_tokens, t, top_p, seed=1)
            out["runs"][f"{t}/{top_p}"] = {"distinct": len(set(answers)), "same as greedy": answers.count(greedy_answer),
                                           "examples": answers[:3]}
            print(f"  temperature {t:.1f}, top-p {top_p}: {len(set(answers))} different answers of {n}")
    out["greedy"] = greedy_answer
    out["same seed, same answers"] = sample(model, tokenizer, SEA, 5, 12, 1.0, seed=7) == sample(model, tokenizer, SEA, 5, 12, 1.0, seed=7)
    out["other seed, same answers"] = sample(model, tokenizer, SEA, 5, 12, 1.0, seed=7) == sample(model, tokenizer, SEA, 5, 12, 1.0, seed=8)
    return out


def first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def in_batches(model, tokenizer, size: int) -> list[list[int]]:
    answers = []
    for start in range(0, len(PROMPTS), size):
        answers += greedy(model, tokenizer, PROMPTS[start:start + size], N_TOKENS)
    return answers


def arithmetic(stage) -> dict:
    """Greedy answers, which involve no randomness at all, under different arithmetic."""
    tokenizer = load_tokenizer()
    f32, b16 = load_model(torch.float32), load_model(torch.bfloat16)
    runs = {
        "float32, alone": stage("float32, alone", lambda: in_batches(f32, tokenizer, 1)),
        "float32, alone, again": stage("float32, alone, again", lambda: in_batches(f32, tokenizer, 1)),
        "float32, in batches of 8": stage("float32, in batches of 8", lambda: in_batches(f32, tokenizer, 8)),
        "bfloat16, alone": stage("bfloat16, alone", lambda: in_batches(b16, tokenizer, 1)),
        "bfloat16, in batches of 2": stage("bfloat16, in batches of 2", lambda: in_batches(b16, tokenizer, 2)),
        "bfloat16, in batches of 8": stage("bfloat16, in batches of 8", lambda: in_batches(b16, tokenizer, 8)),
    }
    comparisons = {
        "float32: the same prompt again": ("float32, alone", "float32, alone, again"),
        "float32: alone, or in a batch of 8": ("float32, alone", "float32, in batches of 8"),
        "bfloat16: alone, or in a batch of 2": ("bfloat16, alone", "bfloat16, in batches of 2"),
        "bfloat16: alone, or in a batch of 8": ("bfloat16, alone", "bfloat16, in batches of 8"),
        "float32 or bfloat16, both alone": ("float32, alone", "bfloat16, alone"),
    }
    out = {"prompts": PROMPTS, "max_new_tokens": N_TOKENS, "comparisons": {}}
    for name, (a, b) in comparisons.items():
        firsts = [first_difference(x, y) for x, y in zip(runs[a], runs[b])]
        out["comparisons"][name] = {"prompts": len(firsts), "changed": sum(f is not None for f in firsts), "first difference": firsts}
        print(f"  {name}: {out['comparisons'][name]['changed']} of {len(firsts)} answers changed; first differences {firsts}")
    examples = []
    for i, d in enumerate(out["comparisons"]["bfloat16: alone, or in a batch of 8"]["first difference"]):
        if d is not None and len(examples) < 3:
            a, b = runs["bfloat16, alone"][i], runs["bfloat16, in batches of 8"][i]
            examples.append({"prompt": PROMPTS[i], "token": d, "same start": tokenizer.decode(a[:d]),
                             "alone": tokenizer.decode(a[d:d + 10]), "in a batch of 8": tokenizer.decode(b[d:d + 10])})
    out["examples"] = examples
    return out


def compute() -> dict:
    RESULTS.mkdir(exist_ok=True)
    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {}

    def stage(name, fn):
        """Run one expensive step once; its result is saved at once, so a rerun resumes here."""
        if name not in partial:
            partial[name] = fn()
            PARTIAL.write_text(json.dumps(partial), encoding="utf-8")
            print(f"  saved stage: {name}")
        return partial[name]

    results: dict = {"machine": machine(), "torch": torch.__version__, "model": MODEL, "revision": REVISION,
                     "threads": torch.get_num_threads()}
    tokenizer = load_tokenizer()
    model = load_model(torch.float32)
    results["generation defaults"] = {k: v for k, v in model.generation_config.to_dict().items()
                                      if k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty")}
    results["first token"] = stage("first token", lambda: {p: first_token(model, tokenizer, p) for p in FIRST_TOKEN_PROMPTS})
    for p, r in results["first token"].items():
        print(f"  {p}: " + ", ".join(f"{t!r} {q:.3f}" for t, q in zip(r["tokens"][:5], r["at temperature"]["1.0"][:5])))
    results["continuations"] = stage("continuations", lambda: continuations(model, tokenizer, FIRST_TOKEN_PROMPTS[0],
                                                                           ["App", "Ban", "Str"]))
    print("  after App, Ban, Str:", {k: [(t, round(q, 3)) for t, q in v] for k, v in results["continuations"].items()})
    results["distinct"] = stage("distinct", lambda: distinct_answers(model, tokenizer))
    del model
    gc.collect()  # free the first copy before the next stage loads two more
    results["arithmetic"] = arithmetic(stage)
    return results


# --- Charts ---------------------------------------------------------------------------------


def _bars(ax, labels, values, colors):
    ax.barh(range(len(labels)), values, color=colors, height=0.7)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.0%}" if v >= 0.005 else "<1%", va="center", fontsize=7.5, color=INK_2)


def draw_sampling(results: dict) -> None:
    prompt = FIRST_TOKEN_PROMPTS[0]
    r = results["first token"][prompt]
    probs = r["at temperature"]["1.0"]
    kept = r["kept by top-p"]["0.8"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    _bars(ax, [repr(t) for t in r["tokens"]], probs, [ACCENT if i < kept else MUTED for i in range(len(probs))])
    ax.set_xlabel("probability of being the first word of the answer")
    ax.set_title(f"\u201c{prompt}\u201d: what the model could say first", fontsize=10)
    ax.text(0.98, 0.05, f"top-p 0.8 keeps the green {kept};\nsampling picks one of them at random,\nweighted by these bars",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ACCENT)
    save(fig, SLUG, "sampling")


def draw_temperature(results: dict) -> None:
    r = results["first token"][FIRST_TOKEN_PROMPTS[1]]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), sharey=True)
    for ax, t in zip(axes, ("0.3", "1.0", "1.5")):
        values = r["at temperature"][t][:8]
        _bars(ax, [repr(x) for x in r["tokens"][:8]], values, [ACCENT] + [MUTED] * (len(values) - 1))
        ax.set_title(f"temperature {t}", fontsize=9.5)
    fig.suptitle(f"\u201c{FIRST_TOKEN_PROMPTS[1]}\u201d: the same scores at three temperatures", fontsize=10)
    fig.tight_layout()
    save(fig, SLUG, "temperature")


def draw_distinct(results: dict) -> None:
    d = results["distinct"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for top_p, color, label in [(1.0, ACCENT, "top-p off"), (0.8, INK_2, "top-p 0.8")]:
        values = [d["runs"][f"{t}/{top_p}"]["distinct"] for t in d["temperatures"]]
        ax.plot(d["temperatures"], values, color=color, linewidth=1.9, marker="o", markersize=4, markeredgecolor="white",
                markeredgewidth=0.8)
        at = d["temperatures"].index(0.2)  # label each line where the two differ most
        left = top_p == 1.0  # the upper line's label goes left of its point, clear of the other line
        ax.text(d["temperatures"][at] + (-0.04 if left else 0.04), values[at], label, fontsize=8.5, color=color,
                va="center", ha="right" if left else "left")
    ax.set_xlim(-0.05, 1.55)
    ax.set_ylim(0, d["n"] + 3)
    ax.set_xlabel("temperature (0 = always the most likely token)")
    ax.set_ylabel(f"different answers out of {d['n']}")
    ax.set_title(f"\u201c{d['prompt']}\u201d, asked {d['n']} times", fontsize=10)
    save(fig, SLUG, "distinct")


def draw_batching(results: dict) -> None:
    comps = results["arithmetic"]["comparisons"]
    names = list(comps)
    changed = [comps[n]["changed"] for n in names]
    totals = [comps[n]["prompts"] for n in names]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    colors = [MUTED if c == 0 else (DANGER if "batch" in n else INK_2) for n, c in zip(names, changed)]
    ax.barh(range(len(names)), changed, color=colors, height=0.6)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals))
    for i, (c, total) in enumerate(zip(changed, totals)):
        ax.text(c + 0.15, i, f"{c} of {total}", va="center", fontsize=8, color=INK)
    ax.set_xlabel(f"greedy answers that changed within {results['arithmetic']['max_new_tokens']} tokens")
    ax.set_title("No randomness at all, and the answer still depends on the arithmetic", fontsize=10)
    save(fig, SLUG, "batching")


if __name__ == "__main__":
    path = RESULTS / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = compute()
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        PARTIAL.unlink()  # finished: the next run starts from scratch
        print(f"  wrote {path.name}")
    draw_sampling(results)
    draw_temperature(results)
    draw_distinct(results)
    draw_batching(results)
