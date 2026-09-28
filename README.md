# ai-demos

The runnable demos behind the AI articles on [chekh.dev](https://chekh.dev): a small
open-weight model, [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
run on an ordinary CPU, and the tokenizers of GPT-2, GPT-4 and GPT-4o. No API keys, no
accounts: every number and chart in the articles comes from a script here.

| Script | Article | What it does |
|---|---|---|
| `tokens.py` | [Tokens: what a model actually reads](https://chekh.dev/writing/tokens-what-a-model-actually-reads/) | Four tokenizers on the same text, a chat template filled in, a BPE tokenizer trained from scratch, the same 1,997 news sentences counted in tokens in sixteen languages, and the model asked to count letters (about a minute) |
| `sampling.py` | [Why the same prompt gives different answers](https://chekh.dev/writing/why-the-same-prompt-gives-different-answers/) | Next-token probabilities, what temperature and top-p do to them, fifty samples at each temperature, and greedy answers alone and in a batch, in float32 and bfloat16 (about 15 minutes; each stage is saved, so an interrupted run resumes) |

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/), and about 6 GB of free memory for
`sampling.py`. The first run downloads the model (about 1 GB, anonymously, from the
Hugging Face Hub), tiktoken's encodings (about 7 MB) and the NTREX-128 text (about 6 MB,
from GitHub). PyTorch comes from its CPU-only index, which keeps the install small.

```sh
git clone https://github.com/thechekh/ai-demos
cd ai-demos
uv sync                          # creates .venv with the pinned versions
uv run pytest                    # the sampler, the BPE tokenizer, and the loop against the library
uv run python tokens.py
uv run python sampling.py
```

Each script prints its numbers as it goes, writes them to `results/<slug>.json`, and
draws its charts to `charts/<slug>/` as SVG; `--charts-only` redraws from the saved
numbers. The articles quote a desktop with an AMD Ryzen 5 3600 and PyTorch on the CPU.
Answers generated greedily are the same on every run of the same machine and settings;
another CPU, another PyTorch version or another thread count can change the arithmetic,
and with it some of the answers, which is what `sampling.py` is about.

## What is in here

- `llm.py` — the model, pinned to one revision; chat formatting; greedy generation
- `tokens.py`, `sampling.py` — the experiments, and the charts they draw
- `tests/` — pytest: the sampler follows temperature and top-p, the BPE tokenizer loses
  no text, the letter test is balanced, and the hand-written generation loop matches the
  library's greedy decoding
- `_common.py`, `paper.mplstyle`, `fonts/` — chart style and typeface (Lora, SIL Open
  Font License)
- `data/` — downloaded text, not committed

## Change something

- `tokens.py` — add a language to `LANGUAGES` (any of the 128 NTREX files, such as
  `"ref.pol"` for Polish), or a tokenizer to `tokenizers()`, and both charts grow
- `sampling.py` — change `SEA` or `TEMPERATURES` and see how fast answers stop repeating;
  add prompts to `PROMPTS`, or load the model in `torch.float16`, and count how many
  greedy answers change

## Licences and data

The code is MIT, see `LICENSE`. Qwen2.5-0.5B-Instruct is released under Apache 2.0.
NTREX-128 ([Federmann et al., 2022](https://aclanthology.org/2022.sumeval-1.4)) is
CC BY-SA 4.0; the scripts download it at run time, and the results contain only counts.
