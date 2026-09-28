"""Tokens: what a model actually reads.

Article: https://chekh.dev/writing/tokens-what-a-model-actually-reads/
Run:     uv run python tokens.py                (about three minutes; the first run also
                                                  downloads the model and the text)
         uv run python tokens.py --charts-only  (redraw from results/)
         uv run pytest tests/test_tokens.py

Four real tokenizers on the same text; a byte-level BPE tokenizer trained from scratch and
traced merge by merge; the NTREX-128 news sentences, the same 1,997 sentences in sixteen
languages, counted in tokens; and a small model asked to count the r's in "strawberry".
"""

import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
from matplotlib.ticker import NullLocator

from _common import ACCENT, INK, INK_2, MUTED, machine, save
from llm import greedy, load_model, load_tokenizer

SLUG = "tokens-what-a-model-actually-reads"
DATA = Path(__file__).parent / "data" / "ntrex"
NTREX = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/468c6b69c7f6a75d31d4743d9daba2af566cc18d/NTREX-128"
LANGUAGES = {
    "English": "src.eng", "German": "ref.deu", "French": "ref.fra", "Spanish": "ref.spa", "Turkish": "ref.tur",
    "Vietnamese": "ref.vie", "Russian": "ref.rus", "Ukrainian": "ref.ukr", "Greek": "ref.ell", "Hebrew": "ref.heb",
    "Arabic": "ref.arb", "Hindi": "ref.hin", "Thai": "ref.tha", "Chinese": "ref.zho-CN", "Japanese": "ref.jpn",
    "Korean": "ref.kor",
}
SPACE = chr(0xB7)  # a middle dot, to make the spaces inside tokens visible


def ntrex(language: str) -> list[str]:
    """The 1,997 NTREX-128 sentences in one language (CC BY-SA 4.0), downloaded once."""
    path = DATA / f"{LANGUAGES[language]}.txt"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{NTREX}/newstest2019-{LANGUAGES[language]}.txt", path)
    return path.read_text(encoding="utf-8").splitlines()


# --- Four real tokenizers -------------------------------------------------------------------


class Tokenizer:
    """One interface over tiktoken's encodings and a Hugging Face tokenizer."""

    def __init__(self, name: str, backend):
        self.name, self.backend = name, backend
        self.hf = not isinstance(backend, tiktoken.Encoding)

    @property
    def vocabulary(self) -> int:
        return len(self.backend) if self.hf else self.backend.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False) if self.hf else self.backend.encode(text)

    def pieces(self, text: str) -> list[str]:
        """The text each token stands for; a token holding part of a character shows as ?."""
        return [self.backend.decode([t]).replace("\ufffd", "?") for t in self.encode(text)]


def tokenizers() -> list[Tokenizer]:
    return [Tokenizer("GPT-2", tiktoken.get_encoding("gpt2")),
            Tokenizer("GPT-4 (cl100k_base)", tiktoken.get_encoding("cl100k_base")),
            Tokenizer("GPT-4o (o200k_base)", tiktoken.get_encoding("o200k_base")),
            Tokenizer("Qwen2.5", load_tokenizer())]


EXAMPLES = ["Tokenizers read pieces, not letters: strawberry, 2026, unbelievable.",
            "strawberry", " strawberry", "Strawberry", "STRAWBERRY", "1234567890", " 2026", "\U0001f44d"]


# --- A byte-level BPE tokenizer from scratch -------------------------------------------------

PRE_SPLIT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+")  # like GPT-2's, simplified


def merge(word: tuple, pair: tuple) -> tuple:
    """Replace every occurrence of `pair` in `word` with the two pieces joined."""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(text: str, merges: int) -> list[tuple[bytes, bytes]]:
    """From single bytes, keep merging the most frequent neighbouring pair."""
    counts = Counter(PRE_SPLIT.findall(text))
    words = {tuple(bytes([b]) for b in w.encode("utf-8")): n
             for w, n in counts.items()}
    learned = []
    for _ in range(merges):
        pairs = Counter()
        for word, n in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] += n
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # ties go to the pair seen first
        learned.append(best)
        words = {merge(word, best): n for word, n in words.items()}
    return learned


def encode_bpe(text: str, learned: list[tuple[bytes, bytes]]) -> list[bytes]:
    """Apply the merges in the order they were learned, word by word."""
    rank = {pair: i for i, pair in enumerate(learned)}
    tokens = []
    for w in PRE_SPLIT.findall(text):
        word = [bytes([b]) for b in w.encode("utf-8")]
        while len(word) > 1:
            best = min(range(len(word) - 1), key=lambda i: rank.get((word[i], word[i + 1]), len(rank)))
            if (word[best], word[best + 1]) not in rank:
                break
            word[best:best + 2] = [word[best] + word[best + 1]]
        tokens.extend(word)
    return tokens


def bpe_experiment(max_merges: int = 2_000) -> dict:
    english, russian = ntrex("English"), ntrex("Russian")
    train, held_out = "\n".join(english[:1_500]), {"English": "\n".join(english[1_500:]),
                                                   "Russian": "\n".join(russian[1_500:])}
    learned = train_bpe(train, max_merges)
    checkpoints = [0, 25, 50, 100, 250, 500, 1_000, 2_000]
    curve = {lang: [len(text) / len(encode_bpe(text, learned[:m])) for m in checkpoints] for lang, text in held_out.items()}
    show = lambda b: b.decode("utf-8", errors="replace").replace(" ", SPACE)  # noqa: E731
    return {"trained on": "NTREX English sentences 1-1,500", "evaluated on": "sentences 1,501-1,997",
            "first merges": [[show(a), show(b), show(a + b)] for a, b in learned[:12]],
            "checkpoints": checkpoints, "characters per token": curve,
            "strawberry after 2,000 merges": [show(t) for t in encode_bpe(" strawberry", learned)],
            "unbelievable after 2,000 merges": [show(t) for t in encode_bpe(" unbelievable", learned)]}


# --- The same text in sixteen languages ------------------------------------------------------


def languages(toks: list[Tokenizer]) -> dict:
    out = {}
    for language in LANGUAGES:
        lines = ntrex(language)
        out[language] = {"characters": sum(len(line) for line in lines),
                         "tokens": {t.name: sum(len(t.encode(line)) for line in lines) for t in toks}}
        if language == "English":
            out[language]["words"] = sum(len(line.split()) for line in lines)
        print(f"  {language:10s} " + "  ".join(f"{n}: {v:>7,}" for n, v in out[language]["tokens"].items()))
    return out


# Five words each with the letter once, twice, three and four times, so a constant guess scores 5 of 20
LETTERS = [("apple", "l"), ("orange", "g"), ("python", "y"), ("window", "i"), ("garden", "r"),
           ("letter", "t"), ("balloon", "o"), ("committee", "t"), ("difficult", "f"), ("coffee", "f"),
           ("strawberry", "r"), ("banana", "a"), ("cheese", "e"), ("parallel", "l"), ("pepper", "p"),
           ("mississippi", "s"), ("assessment", "s"), ("independence", "e"), ("referee", "e"), ("tennessee", "e")]


def count_letters() -> dict:
    """Ask the model to count a letter in 20 words, as written and spelled out letter by letter."""
    tokenizer, model = load_tokenizer(), load_model()
    ask = "How many times does the letter {} appear in {}? Answer with a number only."
    forms = {"the word as written": lambda w: f"the word {w}", "spelled out, s-t-r-a-w...": "-".join}
    out = {"words": [{"word": w, "letter": c, "count": w.count(c), "tokens": len(tokenizer.encode(w, add_special_tokens=False))}
                     for w, c in LETTERS]}
    spelled = "-".join("strawberry")
    out["spelled pieces"] = [tokenizer.decode([i]) for i in tokenizer.encode(spelled, add_special_tokens=False)]
    for form, shape in forms.items():
        prompts = [ask.format(c, shape(w)) for w, c in LETTERS]
        answers = [tokenizer.decode(a) for a in greedy(model, tokenizer, prompts, max_new_tokens=8)]
        numbers = [int(m.group()) if (m := re.search(r"\d+", a)) else None for a in answers]
        for row, answer, number in zip(out["words"], answers, numbers):
            row[form] = {"answer": answer, "correct": number == row["count"]}
        out[form] = sum(row[form]["correct"] for row in out["words"])
        out[f"{form}, answers given"] = dict(Counter(str(n) for n in numbers).most_common())
        print(f"  {form}: {out[form]} of {len(LETTERS)} right; answers given: {out[f'{form}, answers given']}")
    return out


def chat_template() -> dict:
    """What the model receives for a one-word message: the chat template, filled in."""
    tokenizer = load_tokenizer()
    messages = [{"role": "user", "content": "Hi"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"text": text, "tokens": len(tokenizer.encode(text, add_special_tokens=False))}


def compute() -> dict:
    toks = tokenizers()
    results: dict = {"machine": machine(), "tiktoken": tiktoken.__version__,
                     "vocabulary": {t.name: t.vocabulary for t in toks},
                     "encoding for model": {m: tiktoken.encoding_for_model(m).name for m in ("gpt-4", "gpt-4o", "gpt2")}}
    results["pieces"] = {text: {t.name: t.pieces(text) for t in toks} for text in EXAMPLES}
    for text, by in results["pieces"].items():
        print(repr(text).encode("ascii", "backslashreplace").decode(), {n: len(p) for n, p in by.items()})
    results["chat template"] = chat_template()
    print("chat template for 'Hi':", results["chat template"]["tokens"], "tokens")
    results["letter counts"] = count_letters()
    results["bpe"] = bpe_experiment()
    print("first merges:", results["bpe"]["first merges"][:6])
    print("characters per token:", {k: [round(x, 2) for x in v] for k, v in results["bpe"]["characters per token"].items()})
    results["languages"] = languages(toks)
    return results


# --- Charts ---------------------------------------------------------------------------------


def draw_tokens(results: dict) -> None:
    text = EXAMPLES[0]
    pieces = results["pieces"][text]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.3, len(pieces) * 1.3 + 0.6)
    ax.axis("off")
    ax.text(0, len(pieces) * 1.3 + 0.45, text, fontsize=9.5, color=INK, va="center")
    widest = max(sum(len(p) + 1.2 for p in parts) + 0.3 * len(parts) for parts in pieces.values())
    scale = (88 - 21) / widest  # every row fits between the names and the token counts
    for row, (name, parts) in enumerate(pieces.items()):
        y = (len(pieces) - 1 - row) * 1.3
        ax.text(0, y + 0.35, name, fontsize=8.3, color=INK_2, va="center")
        x = 21.0
        for k, part in enumerate(parts):
            label = part.replace(" ", SPACE)
            width = (len(label) + 1.2) * scale
            ax.add_patch(plt.Rectangle((x, y), width, 0.7, facecolor="#dfe5df" if k % 2 == 0 else "#faf9f6",
                                       edgecolor=INK_2, linewidth=0.7))
            ax.text(x + width / 2, y + 0.35, label, ha="center", va="center", fontsize=7.2, color=INK)
            x += width + 0.3 * scale
        ax.text(89.5, y + 0.35, f"{len(parts)} tokens", fontsize=8, color=ACCENT, va="center")
    save(fig, SLUG, "tokens")


def draw_bpe(results: dict) -> None:
    b = results["bpe"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = {"English": ACCENT, "Russian": INK_2}
    for lang, values in b["characters per token"].items():
        ax.plot(b["checkpoints"], values, color=colors[lang], linewidth=1.9, marker="o", markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(b["checkpoints"][-1] * 1.03, values[-1], f"  {lang}", fontsize=8.5, color=colors[lang], va="center")
    ax.set_xlim(-50, 2_450)
    ax.set_ylim(0, max(max(v) for v in b["characters per token"].values()) * 1.15)
    ax.set_xlabel("merges learned from English text")
    ax.set_ylabel("characters per token")
    ax.set_title("A tokenizer trained on English helps English only", fontsize=10)
    save(fig, SLUG, "bpe")


def draw_languages(results: dict) -> None:
    langs = results["languages"]
    names = list(next(iter(langs.values()))["tokens"])
    english = langs["English"]["tokens"]
    order = sorted(langs, key=lambda lang: langs[lang]["tokens"][names[-2]] / english[names[-2]])
    styles = {names[0]: (MUTED, "o"), names[1]: (INK_2, "s"), names[2]: (ACCENT, "D"), names[3]: (INK, "^")}
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for name in names:
        color, marker = styles[name]
        ratios = [langs[lang]["tokens"][name] / english[name] for lang in order]
        ax.scatter(ratios, range(len(order)), color=color, marker=marker, s=26, label=name, zorder=3)
    ax.axvline(1, color=MUTED, linewidth=0.9, linestyle="--")
    ax.set_yticks(range(len(order)), order)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_locator(NullLocator())  # no in-between ticks with their own labels
    ax.set_xlabel("tokens for the same 1,997 sentences, relative to English (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("The same news text costs more tokens in most other languages", fontsize=10)
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
    draw_tokens(results)
    draw_bpe(results)
    draw_languages(results)
