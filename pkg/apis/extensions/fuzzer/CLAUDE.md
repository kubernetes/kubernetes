# Package: fuzzer

## Purpose
Provides fuzzer functions for the extensions API group, used in round-trip testing.

## Key Functions

- **Funcs**: Returns fuzzer functions for extensions types. Currently returns an empty slice.

## Design Notes

- Empty implementation because extensions types are aliases to apps/networking types.
- Fuzzing is handled by the source packages (apps, networking, autoscaling).
