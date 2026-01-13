# Package: ioutils

## Purpose
The `ioutils` package provides I/O utility functions, including a limited writer that caps the amount of data written.

## Key Types/Structs

- **LimitedWriter**: Writer wrapper that limits the amount of data written to N bytes. Each Write call updates N to reflect remaining capacity.

## Key Functions

- **LimitWriter**: Creates a LimitedWriter that writes to an underlying writer but stops with EOF after n bytes.

## Behavior

- Write calls update N to track remaining bytes.
- Returns `io.ErrShortWrite` when the limit is reached.
- If input exceeds remaining capacity, data is truncated and `io.ErrShortWrite` is returned.

## Design Notes

- Mirrors the standard library's `io.LimitReader` pattern but for writers.
- Useful for preventing unbounded writes, such as limiting log output size.
- Thread-safe for individual Write calls (no internal locking, relies on atomic field updates).
