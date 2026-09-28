"""Shared setup for the demos.

Registers Lora so charts match chekh.dev, loads the Paper chart style, and saves each
chart under charts/<article-slug>/. Set CHEKH_DEV_SITE to a checkout of the site to copy
charts straight into its public/images/articles/ as well.
"""

import os
import platform
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
SITE = Path(os.environ["CHEKH_DEV_SITE"]) if os.environ.get("CHEKH_DEV_SITE") else None

for font in (ROOT / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(font)
plt.style.use(ROOT / "paper.mplstyle")

# Paper tokens (chekh.dev DESIGN.md §2.1)
ACCENT = "#2f5d3f"
MUTED = "#b8b8ac"
INK = "#1c1e1a"
INK_2 = "#6f7268"
DANGER = "#8a2f2f"


def save(fig: plt.Figure, slug: str, name: str) -> None:
    """Write charts/<slug>/<name>.svg, with no timestamp so reruns diff cleanly."""
    folder = CHARTS / slug
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT).as_posix()}")
    if SITE:
        target = SITE / "apps" / "web" / "public" / "images" / "articles" / slug / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def plain_numbers(axis) -> None:
    """Log-scale ticks as 1, 10, 1,000 rather than 10⁰, 10¹, 10³."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))


def seconds(value: float) -> str:
    """A duration in the unit that reads best: 42 ns, 830 us, 1.3 s, 2.1 h."""
    for limit, unit, scale in [(1e-6, "ns", 1e9), (1e-3, "us", 1e6), (1, "ms", 1e3), (60, "s", 1), (3600, "min", 1 / 60)]:
        if value < limit:
            number = value * scale
            return f"{number:.2g} {unit}" if number < 10 else f"{number:,.0f} {unit}"
    return f"{value / 3600:.1f} h"


def time_axis(axis) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: seconds(v)))


def time_per_call(fn, min_seconds: float = 0.05, repeats: int = 5) -> float:
    """Seconds per call of `fn`, measured the way `timeit` does it.

    Calls are doubled until one batch takes at least `min_seconds`, so fast functions
    are timed over many calls; then the fastest of `repeats` batches is reported, since
    the fastest run is the one least disturbed by whatever else the machine is doing.
    """
    calls = 1
    while True:
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds:
            break
        calls *= 2
    if elapsed > 1:  # a slow call: fewer repeats keep the whole run reasonable
        repeats = min(repeats, 3)
    best = elapsed
    for _ in range(repeats - 1):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        best = min(best, time.perf_counter() - start)
    return best / calls


def machine() -> str:
    """The line each article quotes, so readers know what produced its timings."""
    return f"Python {platform.python_version()} on {platform.system()}, {os.cpu_count()} logical CPUs"
