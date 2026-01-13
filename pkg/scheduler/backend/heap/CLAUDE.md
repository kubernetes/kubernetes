# Package: heap

## Purpose
Provides a generic thread-safe heap implementation used by the scheduler's priority queue. Supports efficient push, pop, peek, update, and delete operations with O(log n) complexity.

## Key Types

### Heap[T any]
Generic heap struct with configurable:
- **lessFunc**: Comparison function defining heap ordering
- **keyFunc**: Function to extract a unique key from items
- **data**: Internal heapData structure managing the actual heap

### heapData[T any]
Internal storage containing:
- **items**: Map from key to heapItem (for O(1) lookups)
- **queue**: Slice maintaining heap order (for heap operations)

### heapItem[T any]
Wrapper storing an item and its index in the queue for efficient updates.

## Key Functions

- **New[T any](keyFunc, lessFunc)**: Creates a new empty heap
- **Push(obj T)**: Adds an item, replacing if key exists
- **Pop() (T, error)**: Removes and returns the top item
- **Peek() (T, error)**: Returns top item without removing
- **Update(obj T)**: Updates an existing item and reorders
- **Delete(obj T)**: Removes a specific item by key
- **Get(obj T) (T, bool, error)**: Retrieves item by key
- **GetByKey(key string) (T, bool, error)**: Retrieves item by key string
- **List() []T**: Returns all items (unordered)
- **Len() int**: Returns number of items

## Design Pattern
- Combines a map (for O(1) key lookups) with a slice-based heap (for ordering)
- Thread-safe: uses mutex for all operations
- Supports the standard `container/heap` interface via heapData
