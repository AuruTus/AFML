# Information-Code Isolation

Keep **information (knowledge, config, documentation)** separate from **executable code** to avoid coupling and improve maintainability.

## Principles

### Cross-kind isolation (knowledge vs code)
- **Knowledge goes in `knowledge-base/`** — workflow notes, environment setup, conventions. Static, read-only, version-controlled.
- **Code stays in `scripts/`, notebooks, etc.** — executable logic that can be imported, tested, and run.
- **No cross-references from code into `knowledge-base/`** — code must not import or read knowledge files at runtime.

### Within-kind isolation (single responsibility per file)
- **One clear theme per file** — don't lump unrelated topics together even if they're the same kind (e.g., Python environment setup and delivery workflow are both knowledge, but belong in separate files).
- **Explicit, descriptive filenames** — the name alone should convey the file's purpose (e.g., `python-env-setup.md`, `delivery-workflow.md`).
- **Keep file size manageable** — small, focused files are easier to read, maintain, and reason about than monolithic ones.
- **Same rule applies to code** — a module or class should have a single responsibility. Split large modules into smaller, named units rather than one sprawling file.

### Config placement
- **Config belongs in dedicated config files** (e.g., `.python-version`, `.env`), *not* embedded in knowledge docs *or* hardcoded in code.

## Benefits

- Changing a workflow note never breaks a running script.
- New team members can read `knowledge-base/` without touching code.
- Code stays portable — no dependency on non-executable documentation files.
- Small, well-named files are discoverable and self-explanatory.
