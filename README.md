# ai-demos

The runnable demos behind the AI articles on [chekh.dev](https://chekh.dev): a small
open-weight model, [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
run on an ordinary CPU, and the tokenizers of GPT-2, GPT-4 and GPT-4o. No API keys, no
accounts: every number and chart in the articles comes from a script here.

| Script | Article | What it does |
|---|---|---|
| `tokens.py` | [Tokens: what a model actually reads](https://chekh.dev/writing/tokens-what-a-model-actually-reads/) | Four tokenizers on the same text, a chat template filled in, a BPE tokenizer trained from scratch, the same 1,997 news sentences counted in tokens in sixteen languages, and the model asked to count letters (about a minute) |
| `sampling.py` | [Why the same prompt gives different answers](https://chekh.dev/writing/why-the-same-prompt-gives-different-answers/) | Next-token probabilities, what temperature and top-p do to them, fifty samples at each temperature, and greedy answers alone and in a batch, in float32 and bfloat16 (about 15 minutes; each stage is saved, so an interrupted run resumes) |
| `text_metrics.py` | [BLEU, ROUGE and BERTScore: what each one actually counts](https://chekh.dev/writing/bleu-rouge-and-bertscore-what-each-one-actually-counts/) | BLEU, ROUGE, BERTScore, exact match and token F1 written by hand and checked against sacrebleu, rouge-score and bert-score; pairs they score wrongly; and their agreement with experts on SummEval's 1,600 rated summaries (about an hour, most of it BERTScore; progress is saved) |
| `perplexity.py` | [Perplexity: how surprised a model is by your text](https://chekh.dev/writing/perplexity-how-surprised-a-model-is-by-your-text/) | Surprise per token for true and false sentences, perplexity of news, shuffled news, code, random letters, the model's own writing and a journalist's, five languages, and the effect of context (about ten minutes) |
| `robustness.py` | [What Amazon Bedrock's model evaluation actually measures](https://chekh.dev/writing/what-amazon-bedrocks-model-evaluation-actually-measures/) | Bedrock's question-answering recipe run locally: F1 on 200 BoolQ questions, and the drop in F1 under the five prompt perturbations Bedrock's documentation lists, at fmeval's default rates, computed net and per prompt (about twenty minutes; progress is saved) |

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/), and about 6 GB of free memory for
`sampling.py`; run the model scripts one at a time. The first runs download, anonymously:
the model (about 1 GB) and roberta-large for BERTScore (about 1.4 GB) from the Hugging Face
Hub, SummEval and BoolQ from the Hub as well, tiktoken's encodings (about 7 MB) and the
NTREX-128 text (about 6 MB, from GitHub). PyTorch comes from its CPU-only index, which
keeps the install small.

```sh
git clone https://github.com/thechekh/ai-demos
cd ai-demos
uv sync                          # creates .venv with the pinned versions
uv run pytest                    # sampler, BPE tokenizer, metrics against their libraries, perturbations
uv run python tokens.py
uv run python sampling.py
uv run python text_metrics.py
uv run python perplexity.py
uv run python robustness.py
```

Each script prints its numbers as it goes, writes them to `results/<slug>.json`, and
draws its charts to `charts/<slug>/` as SVG; `--charts-only` redraws from the saved
numbers. The articles quote a desktop with an AMD Ryzen 5 3600 and PyTorch on the CPU.
Answers generated greedily are the same on every run of the same machine and settings;
another CPU, another PyTorch version or another thread count can change the arithmetic,
and with it some of the answers, which is what `sampling.py` is about.

## What is in here

- `llm.py` — the model, pinned to one revision; chat formatting; greedy generation
- `tokens.py`, `sampling.py`, `text_metrics.py`, `perplexity.py`, `robustness.py` — the
  experiments, and the charts they draw
- `tests/` — pytest: the sampler follows temperature and top-p, the BPE tokenizer loses
  no text, the letter test is balanced, the hand-written generation loop matches the
  library's greedy decoding, each metric matches its reference library, the surprise
  matches the model's own loss, and the perturbations change only what they claim to
- `_common.py`, `paper.mplstyle`, `fonts/` — chart style and typeface (Lora, SIL Open
  Font License)
- `data/` — downloaded text, not committed

## Change something

- `tokens.py` — add a language to `LANGUAGES` (any of the 128 NTREX files, such as
  `"ref.pol"` for Polish), or a tokenizer to `tokenizers()`, and both charts grow
- `sampling.py` — change `SEA` or `TEMPERATURES` and see how fast answers stop repeating;
  add prompts to `PROMPTS`, or load the model in `torch.float16`, and count how many
  greedy answers change
- `text_metrics.py` — add your own pairs to `PAIRS` and see which metric notices the
  difference that matters to you
- `perplexity.py` — add a pair to `TRUE_FALSE` and see how many bits the false ending costs
- `robustness.py` — change the rates in the perturbation functions (they are
  [fmeval](https://github.com/aws/fmeval)'s defaults), or add a perturbation to
  `PERTURBATIONS`, and see which one the model minds most. After an interrupted run, delete
  `results/*.partial.json` before changing anything, or the stages it finished are reused

## Licences and data

The code is MIT, see `LICENSE`. Qwen2.5-0.5B-Instruct is released under Apache 2.0.
NTREX-128 ([Federmann et al., 2022](https://aclanthology.org/2022.sumeval-1.4)) is
CC BY-SA 4.0; the scripts download it at run time, and the results contain only counts.
SummEval (`mteb/summeval`) is MIT, BoolQ (`google/boolq`) is CC BY-SA 3.0, roberta-large
is MIT and distilroberta-base (used by one test) is Apache 2.0; all are downloaded at run
time, pinned to fixed revisions, and not redistributed here, except BoolQ's yes/no labels
for the 200 sampled questions, which the robustness results list next to the model's
answers (CC BY-SA 3.0, [Clark et al., 2019](https://aclanthology.org/N19-1300/)).
