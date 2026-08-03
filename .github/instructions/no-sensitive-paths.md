# No Sensitive Paths in Committed Content

Generated docs, scripts, and guides must **never** contain absolute host paths
(e.g. `/home/<user>/...`, `/Users/<user>/...`, machine names) that expose the
local machine layout.

## Rules

- Use relative paths (`scripts/...`, `./sample-data/...`) or repo-root-relative
  references in all committed content.
- If an absolute location is unavoidable, use a placeholder like `<repo-root>`.
- Before committing, scan for leaks:

  ```bash
  grep -rn "/home/\|/Users/\|/root/" .
  ```

- Notebooks: Python warning messages embed absolute source paths in stored cell
  outputs (e.g. `/home/<user>/.../triple_barrier_method.py:84: UserWarning: ...`).
  Clear notebook outputs before committing — they regenerate on every run.

## Scope

- Applies to everything authored for git: fix guides, scripts, notebooks,
  knowledge-base docs.
