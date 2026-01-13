# Package: util

## Purpose
Utility functions for working with flowcontrol API types.

## Key Types

- **FlowSchemaSequence**: Slice of FlowSchema pointers that implements sort.Interface for ordering by precedence.

## Sorting Behavior

FlowSchemas are sorted by:
1. MatchingPrecedence (lower precedence values first = higher priority)
2. Name (alphabetically, as tiebreaker)

## Design Notes

- Implements sort.Interface (Len, Less, Swap).
- Used to determine FlowSchema evaluation order.
