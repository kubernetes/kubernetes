# Package: sliceutils

## Purpose
The `sliceutils` package provides sorting utilities for pod and image slices.

## Key Types

- **PodsByCreationTime**: Sortable slice of pods by creation timestamp (ascending order).
- **ByImageSize**: Sortable slice of container images by size (descending order).

## Sorting Behavior

### PodsByCreationTime
- Sorts pods by `CreationTimestamp` in ascending order (oldest first).
- Implements `sort.Interface` (Len, Swap, Less).

### ByImageSize
- Sorts images by `Size` in descending order (largest first).
- Uses image ID as tiebreaker when sizes are equal.
- Implements `sort.Interface` (Len, Swap, Less).

## Design Notes

- Used by kubelet for ordering pods during eviction (oldest first).
- Used by image garbage collection to remove largest images first.
- Simple wrapper types that enable standard library sorting.
