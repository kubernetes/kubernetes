# Package: slice

## Purpose
Provides utility functions for common operations on slices, particularly string slices.

## Key Functions
- `CopyStrings()` - Creates a copy of a string slice
- `SortStrings()` - Sorts a string slice in place and returns it for chaining
- `ContainsString()` - Checks if slice contains a string, with optional modifier function
- `RemoveString()` - Creates new slice without matching string(s)

## Design Patterns
- Copy functions handle nil input by returning nil
- Modifier function allows custom comparison (e.g., case-insensitive)
- SortStrings returns same slice for method chaining
- RemoveString returns nil for empty result (not empty slice)
- Non-mutating where possible (RemoveString creates new slice)
