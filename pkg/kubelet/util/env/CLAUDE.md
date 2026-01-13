# Package: env

## Purpose
The `env` package provides a strict environment file parser using a subset of POSIX shell syntax, designed for parsing environment variable files safely.

## Key Functions

- **ParseEnv**: Parses an environment file and returns the value for a specified key. Enforces strict format requirements.

## Format Requirements

- Values must be enclosed in single quotes: `VAR='value'`
- Content within single quotes is preserved literally (no escape sequences)
- Multi-line values are supported (newlines within quotes preserved)
- Inline comments after closing quote are supported: `VAR='value' # comment`
- Leading whitespace before variable name is ignored
- Blank lines and comment lines (starting with #) are ignored
- Whitespace before `=` is invalid: `VAR = 'value'` is rejected
- Whitespace after `=` results in empty assignment: `VAR= 'value'` assigns empty string

## Design Notes

- Implements strict parsing to avoid shell injection vulnerabilities.
- Matches bash behavior for edge cases like whitespace handling.
- Returns empty string (not error) if key is not found.
- Returns error for malformed files (unclosed quotes, missing `=`, etc.).
