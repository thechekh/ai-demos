"""BLEU, ROUGE and BERTScore: what each one actually counts.

Article: https://chekh.dev/writing/bleu-rouge-and-bertscore-what-each-one-actually-counts/
Run:     uv run python text_metrics.py                (about half an hour on a CPU: BERTScore
                                                       embeds 2,700 summaries with roberta-large,
                                                       a 1.4 GB download; progress is saved)
         uv run python text_metrics.py --charts-only  (redraw from results/)
         uv run pytest tests/test_text_metrics.py

BLEU, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, exact match and token F1, each written out in a
few lines and checked against the reference libraries (sacrebleu, rouge-score, bert-score);
pairs of sentences where the metrics and common sense disagree; and SummEval, 1,600 news
summaries rated by experts, to measure how far each metric agrees with people.
"""

import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from _common import ACCENT, DANGER, INK, INK_2, MUTED, machine, save

SLUG = "bleu-rouge-and-bertscore-what-each-one-actually-counts"
RESULTS = Path(__file__).parent / "results"
PARTIAL = RESULTS / f"{SLUG}.partial.json"
SUMMEVAL = ("mteb/summeval", "data/test-00000-of-00001-35901af5f6649399.parquet",
            "bfc121155064afa2d81b5505682ffc0d96f4334c")
BERT_MODEL, BERT_LAYER = "roberta-large", 17  # bert-score's default model and layer for English
BERT_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"  # pinned: the same weights on every run


# --- Overlap metrics ------------------------------------------------------------------------

WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    """Lowercase words and numbers, punctuation dropped: what the overlap metrics compare."""
    return WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: list[str], references: list[list[str]],
         smooth: float = 1.0) -> float:
    """Clipped 1- to 4-gram precision, geometric mean, times a length penalty."""
    log_precision = 0.0
    for n in range(1, 5):
        counts = ngrams(candidate, n)
        allowed = Counter()
        for reference in references:
            allowed |= ngrams(reference, n)  # the most any reference uses it
        matched = sum(min(c, allowed[g]) for g, c in counts.items())
        total = sum(counts.values())
        if n > 1:  # add-k smoothing: a missing 4-gram must not zero it
            matched, total = matched + smooth, total + smooth
        if matched == 0:
            return 0.0
        log_precision += math.log(matched / total) / 4
    closest = min((abs(len(r) - len(candidate)), len(r)) for r in references)[1]
    short = len(candidate) < closest
    brevity = math.exp(1 - closest / len(candidate)) if short else 1.0
    return brevity * math.exp(log_precision)


def f_measure(overlap: int, candidate_size: int, reference_size: int) -> float:
    if overlap == 0:
        return 0.0
    precision, recall = overlap / candidate_size, overlap / reference_size
    return 2 * precision * recall / (precision + recall)


def rouge_n(candidate: list[str], reference: list[str], n: int) -> float:
    """ROUGE-N as an F-measure of the n-grams the two texts share."""
    cand, ref = ngrams(candidate, n), ngrams(reference, n)
    overlap = sum((cand & ref).values())
    return f_measure(overlap, sum(cand.values()), sum(ref.values()))


def lcs(a: list[str], b: list[str]) -> int:
    """Longest common subsequence: shared words, in order, gaps allowed."""
    row = [0] * (len(b) + 1)
    for x in a:
        diagonal = 0
        for j, y in enumerate(b, 1):
            best = diagonal + 1 if x == y else max(row[j], row[j - 1])
            diagonal, row[j] = row[j], best
    return row[-1]


def rouge_l(candidate: list[str], reference: list[str]) -> float:
    return f_measure(lcs(candidate, reference), len(candidate), len(reference))


# --- BERTScore ------------------------------------------------------------------------------


class BertScorer:
    """BERTScore by hand: each token's vector in context, matched to its most similar."""

    def __init__(self, model_name: str = BERT_MODEL, layer: int = BERT_LAYER, revision: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).eval()
        self.layer, self.cache = layer, {}

    @torch.no_grad()
    def vectors(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit vectors for every token, and a mask that is False for the start and end tokens."""
        if text not in self.cache:
            ids = self.tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=510,
                                 add_prefix_space=True, return_special_tokens_mask=True)
            special = ids.pop("special_tokens_mask")[0].bool()
            hidden = self.model(**ids, output_hidden_states=True).hidden_states[self.layer][0]
            self.cache[text] = (F.normalize(hidden, dim=-1), ~special)
        return self.cache[text]

    def score(self, candidate: str, reference: str) -> tuple[float, float, float]:
        c, c_words = self.vectors(candidate)
        r, r_words = self.vectors(reference)
        similarity = c @ r.T  # cosine similarity, every pair of tokens
        precision = similarity.max(dim=1).values[c_words].mean().item()
        recall = similarity.max(dim=0).values[r_words].mean().item()
        return precision, recall, 2 * precision * recall / (precision + recall)

    def best_matches(self, candidate: str, reference: str) -> list[list]:
        """For the chart: each candidate word piece, its best reference piece, and the similarity."""
        c, c_words = self.vectors(candidate)
        r, _ = self.vectors(reference)
        c_ids = self.tokenizer(candidate.strip(), add_prefix_space=True)["input_ids"]
        r_ids = self.tokenizer(reference.strip(), add_prefix_space=True)["input_ids"]
        similarity = c @ r.T
        out = []
        for i in range(len(c_ids)):
            if c_words[i]:
                j = int(similarity[i].argmax())
                out.append([self.tokenizer.decode([c_ids[i]]).strip(), self.tokenizer.decode([r_ids[j]]).strip(),
                            float(similarity[i, j])])
        return out


# --- Short answers: exact match and token F1 -------------------------------------------------


def normalize_answer(text: str) -> str:
    """SQuAD's normalisation: lowercase, drop punctuation and a/an/the, squeeze spaces."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))


def token_f1(prediction: str, truth: str) -> float:
    """Harmonic mean of precision and recall over the words both answers use."""
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(truth).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    return f_measure(common, len(pred), len(gold))


# --- Agreement with people -------------------------------------------------------------------


def ranks(values) -> np.ndarray:
    """Ranks from 1, with tied values sharing the average of their ranks."""
    values = np.asarray(values, dtype=float)
    order = values.argsort(kind="stable")
    r = np.empty(len(values))
    r[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):
        tied = values == v
        r[tied] = r[tied].mean()
    return r


def spearman(x, y) -> float:
    """Spearman's rank correlation: Pearson's correlation of the two rank lists."""
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


# --- Experiments ----------------------------------------------------------------------------

PAIRS = [  # (what the pair shows, reference, candidate)
    ("identical", "The drug is safe for children.", "The drug is safe for children."),
    ("the same meaning, other words", "The drug is safe for children.", "Kids can take this medicine without risk."),
    ("the opposite meaning", "The drug is safe for children.", "The drug is not safe for children."),
    ("a different fact", "The drug is safe for children.", "The drug is safe for adults."),
    ("the words shuffled", "The drug is safe for children.", "Children for safe is drug the."),
    ("right, but short", "The drug is safe for children.", "Safe."),
    ("unrelated", "The drug is safe for children.", "Stock markets fell sharply on Monday."),
]
ANSWERS = [("Paris", "Paris"), ("Paris, France", "Paris"), ("The city of Paris", "Paris"), ("It is not Paris", "Paris"),
           ("Lyon", "Paris")]
CHART_PAIR = ("The drug is safe for children.", "The medicine is safe for kids.")


def surprises(scorer: BertScorer) -> list[dict]:
    rows = []
    for label, reference, candidate in PAIRS:
        c, r = words(candidate), words(reference)
        rows.append({"pair": label, "reference": reference, "candidate": candidate, "BLEU": bleu(c, [r]),
                     "ROUGE-1": rouge_n(c, r, 1), "ROUGE-2": rouge_n(c, r, 2), "ROUGE-L": rouge_l(c, r),
                     "BERTScore": scorer.score(candidate.lower(), reference.lower())[2]})
        print(f"  {label:32s} " + "  ".join(f"{k} {v:.2f}" for k, v in rows[-1].items() if isinstance(v, float)))
    return rows


def summeval_rows() -> list[dict]:
    path = hf_hub_download(SUMMEVAL[0], SUMMEVAL[1], repo_type="dataset", revision=SUMMEVAL[2])
    return pq.read_table(path).to_pylist()


def score_summeval(scorer: BertScorer, stage) -> dict:
    """Every metric for all 1,600 summaries, against the 11 human-written references each."""
    per_summary = []
    for doc, row in enumerate(summeval_rows()):
        def one_document(row=row):
            refs = [r.lower() for r in row["human_summaries"]]
            ref_words = [words(r) for r in refs]
            out = []
            for k, summary in enumerate(row["machine_summaries"]):
                s = summary.lower()
                sw = words(s)
                out.append({"BLEU": bleu(sw, ref_words),
                            "ROUGE-1": max(rouge_n(sw, rw, 1) for rw in ref_words),
                            "ROUGE-2": max(rouge_n(sw, rw, 2) for rw in ref_words),
                            "ROUGE-L": max(rouge_l(sw, rw) for rw in ref_words),
                            "BERTScore": max(scorer.score(s, r)[2] for r in refs),
                            "length": len(sw),
                            **{dim: row[dim][k] for dim in ("relevance", "consistency", "coherence", "fluency")}})
            return out
        per_summary += stage(f"summeval {doc:03d}", one_document)
        if doc % 10 == 9:
            print(f"  scored {len(per_summary):,} summaries")
    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "length"]
    dims = ["relevance", "consistency", "coherence", "fluency"]
    return {"summaries": len(per_summary), "references per summary": 11, "metrics": metrics, "dimensions": dims,
            "spearman": {m: {d: spearman([s[m] for s in per_summary], [s[d] for s in per_summary]) for d in dims}
                         for m in metrics},
            "mean": {m: float(np.mean([s[m] for s in per_summary])) for m in metrics + dims}}


def compute() -> dict:
    RESULTS.mkdir(exist_ok=True)
    partial = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {}

    def stage(name, fn):
        if name not in partial:
            partial[name] = fn()
            PARTIAL.write_text(json.dumps(partial), encoding="utf-8")
        return partial[name]

    scorer = BertScorer(revision=BERT_REVISION)
    results: dict = {"machine": machine(), "bertscore model": f"{BERT_MODEL}, layer {BERT_LAYER}",
                     "summeval revision": SUMMEVAL[2]}
    results["surprises"] = surprises(scorer)
    results["answers"] = [{"prediction": p, "truth": t, "exact match": exact_match(p, t), "token F1": token_f1(p, t)}
                          for p, t in ANSWERS]
    for a in results["answers"]:
        print(f"  {a['prediction']!r:20} vs {a['truth']!r}: EM {a['exact match']:.0f}, F1 {a['token F1']:.2f}")
    results["chart pair"] = {"reference": CHART_PAIR[0], "candidate": CHART_PAIR[1],
                             "matches": scorer.best_matches(CHART_PAIR[1].lower(), CHART_PAIR[0].lower()),
                             "BLEU": bleu(words(CHART_PAIR[1]), [words(CHART_PAIR[0])]),
                             "ROUGE-1": rouge_n(words(CHART_PAIR[1]), words(CHART_PAIR[0]), 1),
                             "BERTScore": scorer.score(CHART_PAIR[1].lower(), CHART_PAIR[0].lower())[2]}
    results["summeval"] = score_summeval(scorer, stage)
    for m, by in results["summeval"]["spearman"].items():
        print(f"  {m:10s} " + "  ".join(f"{d} {v:+.2f}" for d, v in by.items()))
    return results


# --- Charts ---------------------------------------------------------------------------------


def draw_matching(results: dict) -> None:
    """The same pair, scored by exact word matches and by BERTScore's soft matches."""
    pair = results["chart pair"]
    ref, cand = words(pair["reference"]), words(pair["candidate"])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.4))
    for ax in axes:
        ax.set_xlim(-0.3, 8.6)
        ax.set_ylim(-0.4, 2.0)
        ax.axis("off")

    def row(ax, tokens, y, highlight):
        xs = []
        for i, t in enumerate(tokens):
            x = 0.9 + i * 1.02
            xs.append(x)
            ax.add_patch(plt.Rectangle((x - 0.46, y - 0.2), 0.92, 0.4, facecolor="#dfe5df" if highlight(t) else "#faf9f6",
                                       edgecolor=INK_2, linewidth=0.8))
            ax.text(x, y, t, ha="center", va="center", fontsize=8.6, color=INK)
        return xs

    ax = axes[0]
    ax.text(-0.3, 1.85, "BLEU and ROUGE: exact words only", fontsize=9.3, color=INK, va="top")
    ax.text(-0.3, 1.2, "reference", fontsize=8, color=INK_2, va="center")
    ax.text(-0.3, 0.2, "candidate", fontsize=8, color=INK_2, va="center")
    rx = row(ax, ref, 1.2, lambda t: t in cand)
    cx = row(ax, cand, 0.2, lambda t: t in ref)
    for i, t in enumerate(cand):
        if t in ref:
            ax.annotate("", (rx[ref.index(t)], 1.0), (cx[i], 0.4), arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": 1.1})
    ax.text(8.55, 0.2, f"ROUGE-1 {pair['ROUGE-1']:.2f}\nBLEU {pair['BLEU']:.2f}", fontsize=8.3, color=ACCENT, ha="right", va="center")

    ax = axes[1]
    ax.text(-0.3, 1.85, "BERTScore: every candidate word to its most similar reference word", fontsize=9.3, color=INK, va="top")
    rx = row(ax, ref, 1.2, lambda t: True)
    cx = row(ax, cand, 0.2, lambda t: True)
    for i, (c, r, sim) in enumerate(pair["matches"][:len(cand)]):
        j = ref.index(r) if r in ref else None
        if j is not None:
            ax.annotate("", (rx[j], 1.0), (cx[i], 0.4), arrowprops={"arrowstyle": "-", "color": ACCENT if c != r else INK_2,
                                                                  "linewidth": 1.1})
            ax.text((rx[j] + cx[i]) / 2 + 0.05, 0.7, f"{sim:.2f}", fontsize=7, color=ACCENT if c != r else INK_2, ha="left")
    ax.text(8.55, 0.2, f"BERTScore {pair['BERTScore']:.2f}", fontsize=8.3, color=ACCENT, ha="right", va="center")
    fig.tight_layout()
    save(fig, SLUG, "matching")


def draw_correlation(results: dict) -> None:
    s = results["summeval"]
    metrics, dims = s["metrics"], s["dimensions"]
    colors = {"relevance": ACCENT, "consistency": INK, "coherence": INK_2, "fluency": MUTED}
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    height = 0.19
    for k, dim in enumerate(dims):
        values = [s["spearman"][m][dim] for m in metrics]
        ax.barh([i + (k - 1.5) * height for i in range(len(metrics))], values, height=height, color=colors[dim], label=dim)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(range(len(metrics)), metrics)
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 0.6)
    ax.set_xlabel("Spearman correlation with expert ratings, 1,600 summaries")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("How far each metric agrees with people", fontsize=10)
    save(fig, SLUG, "correlation")


if __name__ == "__main__":
    path = RESULTS / f"{SLUG}.json"
    if "--charts-only" in sys.argv:
        results = json.loads(path.read_text(encoding="utf-8"))
    else:
        results = compute()
        path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        PARTIAL.unlink()
        print(f"  wrote {path.name}")
    draw_matching(results)
    draw_correlation(results)
