---
description: "Use when configuring or selecting a Python environment/interpreter in this repo. If a .python-version file exists, prefer the Python version it pins before any other environment detection."
---
# Python Version Precedence

When setting up or selecting a Python environment/interpreter in this repo,
check for a `.python-version` file first and honor it.

## Rules

- If `.python-version` exists at the workspace root, use the version it pins
  (pyenv format, e.g. `3.12.4`) as the required interpreter.
- Prefer an interpreter, venv, or pyenv environment that matches that exact
  version before falling back to any other detection (default `python`,
  system Python, or whatever `configure_python_environment` would pick).
- Do not create, activate, or install into an environment whose Python version
  differs from `.python-version` unless the user explicitly asks.
- When running Python commands, use the matching interpreter (e.g.
  `pyenv exec python` or the venv built from that version) instead of the
  bare `python` command.
