---
inclusion: always
---

# Python Best Practices

## Formatting & Style
- Follow PEP 8: 4-space indentation, max 88 chars per line (Black-compatible)
- Use `ruff` for linting and formatting checks
- One blank line between methods, two between top-level definitions
- No trailing whitespace, files end with a newline

## Imports
- All imports at the top of the file, never inline
- Order: stdlib → third-party → local, separated by blank lines
- No wildcard imports (`from x import *`)
- Remove unused imports

## Docstrings
- Write docstrings in English
- Use triple double-quotes for all public modules, functions, classes, and methods
- One-liner for simple functions; multi-line (summary + args/returns) for complex ones
- No redundant docstrings that just restate the function name

## Comments
- No unnecessary comments — code should be self-explanatory
- Only comment on non-obvious logic or business decisions
- Never leave commented-out code in the codebase

## Functions & Methods
- Keep functions small and focused: one responsibility per function
- Prefer flat over nested logic; use early returns to reduce indentation
- Max ~20 lines per function; if longer, consider splitting
- Use descriptive names — no abbreviations unless universally known (`df`, `idx`)

## DRY Principles
- Extract repeated logic into helper functions
- Avoid copy-paste code; if you write something twice, abstract it
- Reuse constants instead of magic numbers/strings

## No Over-Engineering
- Don't add abstractions until they're needed
- Prefer simple functions over classes when state isn't required
- Avoid premature optimization
- No unnecessary design patterns or layers of indirection
