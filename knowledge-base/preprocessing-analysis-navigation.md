# Preprocessing analysis navigation

The current pipeline review is [docs/data-preprocessing-pipeline-analysis.md](../docs/data-preprocessing-pipeline-analysis.md). It records source locations, actual input schemas, deterministic failure examples, baseline counts, a reproducible price-path comparison, and proposed repair order.

The follow-up [prototype design](../docs/quant-preprocessing-prototype-design.md) proposes implementation stages, function coverage, data contracts, a Polars/pandas boundary, and shared incremental state for backtests and online extraction. The user confirmed those two targets and potential history of hundreds of GB; GPU work is for future model training and is deferred. The APIs and remaining market/latency defaults are a discussion draft, not an implemented specification.

The user subsequently selected spot-first scope and `toy-quant` as the future implementation repository. The design now distinguishes market-data adapters from preprocessing/historical labeling, accounts for the existing SDK-based OKX broker, and includes a proposed delegation/review workflow. The user will move the documents; do not infer authorization to start coding from this design discussion.

For subsequent work:

- Honor `.python-version` (`mlquant`) using `pyenv exec python`. Verify the imported `cqrlib` resolves to the intended sibling checkout before comparing results.
- Notebook cell references in the analysis count markdown and code cells from one. Inspect the current working tree; existing notebook edits may differ from HEAD.
- `sample-data/*_bars.csv` and ordinary `*_bars.txt` are different datasets with different activity-column names. The `*_bars copy.txt` files matched the legacy CSV files at review time.
- Keep the full bar price path separate from sparse event starts. Label exits and concurrency require the observation path, not only sampled feature rows.
- Do not assume a chronological split or the present `PurgedKFold` implementation isolates overlapping label intervals. The report contains counterexamples and acceptance criteria.
- The delivery guide's referenced `python-env-setup.md` was not present at review time; `.python-version` and the Python precedence instruction provided the available environment guidance.

This note is documentation only and must not become a runtime dependency.
