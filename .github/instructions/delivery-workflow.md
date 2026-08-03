# Delivery Workflow

Follow a structured **plan → execute → validate → conclude** cycle for every task.

## Planning
- Understand the request and read relevant code/files first.
- Break the work into clear steps (use a todo list for multi-step tasks).
- Identify dependencies and potential issues (e.g., script bugs, missing data).

## Execution
- Make minimal, targeted changes — don't over-engineer.
- Fix issues as they surface (e.g., column header mismatches).
- Run scripts using the correct pyenv environment (see `python-env-setup.md`).

## Validation
- Verify outputs programmatically (e.g., `pd.read_csv` to confirm data loads correctly).
- Check row counts, column names, index types, and date ranges match expectations.
- Fix any validation failures before concluding.

## Conclusion
- Summarise what was done, what files were created/modified, and key metrics.
- Confirm the deliverable is ready for the next step (e.g., notebook execution).

## Knowledge Preservation
- If any skill, pattern, or information from this task could be useful in **future sessions or for other agents**, record it in `knowledge-base/`.
- Create a focused file with a clear, explicit name (see `information-code-isolation.md`).
- This ensures reusable knowledge persists beyond the current task.
