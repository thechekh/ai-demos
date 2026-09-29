"""What Amazon Bedrock's model evaluation actually measures.

Article: https://chekh.dev/writing/what-amazon-bedrocks-model-evaluation-actually-measures/
Run:     uv run python robustness.py                (about twenty minutes on a CPU)
         uv run python robustness.py --charts-only  (redraw from results/)
         uv run pytest tests/test_robustness.py

Bedrock's recipe for question answering, run locally on a small open-weight model with no
AWS account: accuracy as token F1 on BoolQ, one of Bedrock's built-in datasets, and
robustness as the drop in F1 when each prompt is perturbed in the five ways Bedrock's
documentation lists: all lower case, keyboard typos, numbers written as words, random upper
case, and whitespace added or removed. The documentation gives no rates, so these are the
defaults of AWS's open-source evaluation library, fmeval (github.com/aws/fmeval), and, as in
fmeval, the rewrites change the passage and the question while the instruction around them
stays as written. The drop is computed two ways: between the two average F1s, and prompt by
prompt with the sign dropped, the way fmeval computes it.
"""

import json
import math
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from _common import ACCENT, INK, INK_2, MUTED, machine, save
from llm import MODEL, REVISION, greedy, load_model, load_tokenizer
from text_metrics import normalize_answer, token_f1

SLUG = "what-amazon-bedrocks-model-evaluation-actually-measures"
RESULTS = Path(__file__).parent / "results"
PARTIAL = RESULTS / f"{SLUG}.partial.json"
BOOLQ = ("google/boolq", "data/validation-00000-of-00001.parquet", "35b264d03638db9f4ce671b711558bf7ff0f80d5")
N_QUESTIONS, MAX_WORDS = 200, 150


# --- Five perturbations, the kinds Bedrock's documentation lists -----------------------------
# The rates are fmeval's defaults: typos 0.1, upper case 0.1, spaces removed 0.1 and added 0.05.

KEYS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEAR = {}
for r, row in enumerate(KEYS):
    for c, key in enumerate(row):
        NEAR[key] = "".join(KEYS[rr][cc] for rr in (r - 1, r, r + 1) for cc in (c - 1, c, c + 1)
                            if 0 <= rr < 3 and 0 <= cc < len(KEYS[rr]) and (rr, cc) != (r, c))

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + number_words(n % 100))
    return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))


def lower_case(text: str, rng: random.Random | None = None) -> str:
    return text.lower()


def keyboard_typos(text: str, rng: random.Random, rate: float = 0.1) -> str:
    """Now and then, a letter becomes one of the keys around it."""
    out = []
    for ch in text:
        near = NEAR.get(ch.lower())
        if near and rng.random() < rate:
            typo = rng.choice(near)
            ch = typo.upper() if ch.isupper() else typo
        out.append(ch)
    return "".join(out)


def numbers_as_words(text: str, rng: random.Random | None = None) -> str:
    return re.sub(r"\b\d{1,4}\b", lambda m: number_words(int(m.group())), text)


def random_upper_case(text: str, rng: random.Random, rate: float = 0.1) -> str:
    return "".join(ch.upper() if rng.random() < rate else ch for ch in text)


def whitespace(text: str, rng: random.Random, remove: float = 0.1, add: float = 0.05) -> str:
    """Drop some spaces between words, and put a few in the middle of words."""
    out = []
    for ch in text:
        if ch == " " and rng.random() < remove:
            continue
        out.append(ch)
        if ch != " " and rng.random() < add:
            out.append(" ")
    return "".join(out)


ILLUSTRATION = "Did the New York Yankees win the 1996 World Series in 6 games, beating the Atlanta Braves?"

PERTURBATIONS = {"all lower case": lower_case, "keyboard typos": keyboard_typos, "numbers as words": numbers_as_words,
                 "random upper case": random_upper_case, "whitespace added or removed": whitespace}


# --- The evaluation --------------------------------------------------------------------------


def boolq(n: int = N_QUESTIONS, seed: int = 0) -> list[dict]:
    """n BoolQ validation questions, passages cut to their first MAX_WORDS words."""
    path = hf_hub_download(BOOLQ[0], BOOLQ[1], repo_type="dataset", revision=BOOLQ[2])
    rows = pq.read_table(path).to_pylist()
    picked = random.Random(seed).sample(rows, n)
    return [{"passage": " ".join(r["passage"].split()[:MAX_WORDS]), "question": r["question"].rstrip("?") + "?",
             "answer": "yes" if r["answer"] else "no"} for r in picked]


def prompt(item: dict) -> str:
    return f"{item['passage']}\n\nQuestion: {item['question']}\nAnswer yes or no."


def perturbed_prompt(item: dict, perturb, rng: random.Random) -> str:
    """Rewrite the passage and the question; the instruction stays as written, as in fmeval."""
    return prompt({**item, "passage": perturb(item["passage"], rng), "question": perturb(item["question"], rng)})


def first_word(reply: str) -> str:
    words = normalize_answer(reply).split()
    return words[0] if words else ""


def answer(model, tokenizer, prompts: list[str], batch: int = 8) -> list[str]:
    replies = []
    for i in range(0, len(prompts), batch):
        replies += [tokenizer.decode(r) for r in greedy(model, tokenizer, prompts[i:i + batch], max_new_tokens=4)]
    return [first_word(r) for r in replies]


def sign_test(worse: int, better: int) -> float:
    """How often fair coin tosses would split the changed answers at least this unevenly."""
    n = worse + better
    tail = sum(math.comb(n, k) for k in range(min(worse, better) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def evaluate(stage) -> dict:
    tokenizer, model = load_tokenizer(), load_model()
    items = boolq()
    original = [prompt(it) for it in items]
    truth = [it["answer"] for it in items]
    runs = {"original": stage("original", lambda: answer(model, tokenizer, original))}
    for name, perturb in PERTURBATIONS.items():
        rng = random.Random(name)  # the same perturbations on every run
        perturbed = [perturbed_prompt(it, perturb, rng) for it in items]
        runs[name] = stage(name, lambda perturbed=perturbed: answer(model, tokenizer, perturbed))
        print(f"  {name}: done")
    f1 = {name: sum(token_f1(a, t) for a, t in zip(answers, truth)) / len(truth) for name, answers in runs.items()}
    base = f1["original"]
    out = {"questions": len(items), "passage words, at most": MAX_WORDS, "yes share": truth.count("yes") / len(truth),
           "F1": f1, "answers given, original": {w: runs["original"].count(w) for w in set(runs["original"])},
           "robustness": {}}
    right = [token_f1(a, t) == 1 for a, t in zip(runs["original"], truth)]
    for name in PERTURBATIONS:
        flipped = sum(a != b for a, b in zip(runs["original"], runs[name]))
        now = [token_f1(a, t) == 1 for a, t in zip(runs[name], truth)]
        lost = sum(r and not n for r, n in zip(right, now))
        gained = sum(n and not r for r, n in zip(right, now))
        # fmeval's way: each prompt's own change, sign dropped, then averaged
        per_prompt = sum(abs(token_f1(a, t) - token_f1(b, t))
                         for a, b, t in zip(runs["original"], runs[name], truth)) / len(truth)
        out["robustness"][name] = {"F1": f1[name], "delta F1, net": base - f1[name],
                                   "robustness score, net (%)": (base - f1[name]) / base * 100,
                                   "delta F1, per prompt": per_prompt,
                                   "robustness score, per prompt (%)": per_prompt / base * 100,
                                   "answers changed": flipped, "right to wrong": lost, "wrong to right": gained,
                                   "sign test p": sign_test(lost, gained)}
        print(f"  {name:28s} F1 {f1[name]:.3f}  score net {(base - f1[name]) / base * 100:+.1f}%, "
              f"per prompt {per_prompt / base * 100:.1f}%  changed {flipped} (right to wrong {lost}, "
              f"wrong to right {gained}, sign test p {sign_test(lost, gained):.2f})")
    all_perturbed = [f1[n] for n in PERTURBATIONS]
    out["robustness, all five, net (%)"] = (base - sum(all_perturbed) / len(all_perturbed)) / base * 100
    out["robustness, all five, per prompt (%)"] = sum(out["robustness"][n]["delta F1, per prompt"]
                                                      for n in PERTURBATIONS) / len(PERTURBATIONS) / base * 100
    out["answers"] = {"truth": truth, **runs}
    return out


def compute() -> dict:
    RESULTS.mkdir(exist_ok=True)
    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {}

    def stage(name, fn):
        if name not in partial:
            partial[name] = fn()
            PARTIAL.write_text(json.dumps(partial), encoding="utf-8")
        return partial[name]

    results = {"machine": machine(), "torch": torch.__version__, "model": MODEL, "revision": REVISION,
               "boolq revision": BOOLQ[2]}
    results["illustration"] = {"original": ILLUSTRATION, **{name: perturb(ILLUSTRATION, random.Random(0))
                                                            for name, perturb in PERTURBATIONS.items()}}
    results["qa"] = evaluate(stage)
    print(f"  F1 on the original prompts: {results['qa']['F1']['original']:.3f}")
    return results


def draw_robustness(results: dict) -> None:
    qa = results["qa"]
    names = list(qa["robustness"])
    ways = [("robustness score, net (%)", "net: the drop in the average F1", INK_2),
            ("robustness score, per prompt (%)", "per prompt, sign dropped, as fmeval computes it", ACCENT)]
    top = max(qa["robustness"][n][key] for n in names for key, _, _ in ways)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for k, (key, label, color) in enumerate(ways):
        values = [qa["robustness"][n][key] for n in names]
        ys = [i + (k - 0.5) * 0.38 for i in range(len(names))]
        ax.barh(ys, values, height=0.36, color=color, label=label)
        for y, v in zip(ys, values):
            ax.text(max(v, 0) + top * 0.015, y, f"{v:.1f}%".replace("-", "\u2212"), va="center", fontsize=7.8,
                    color=INK)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(right=top * 1.15)
    ax.set_xlabel("robustness score: the drop in F1, as a percentage of F1 (lower is better)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
    ax.set_title(f"BoolQ, {qa['questions']} questions: the same answers, scored two ways", fontsize=10, pad=26)
    save(fig, SLUG, "robustness")


if __name__ == "__main__":
    path = RESULTS / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = compute()
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        PARTIAL.unlink()
        print(f"  wrote {path.name}")
    draw_robustness(results)
