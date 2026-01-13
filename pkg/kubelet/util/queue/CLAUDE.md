# Package: queue

## Purpose
The `queue` package provides a time-based work queue where items become ready to process after a specified delay.

## Key Interfaces

- **WorkQueue**: Interface for a delayed work queue.
  - `GetWork()`: Dequeues and returns all items whose timestamps have expired.
  - `Enqueue(item, delay)`: Inserts a new item or overwrites existing with a new delay.

## Key Types/Structs

- **basicWorkQueue**: Simple implementation using a map of UID to timestamp.

## Key Functions

- **NewBasicWorkQueue**: Creates a new basic WorkQueue with the provided clock.

## Behavior

- Items are keyed by `types.UID`.
- Enqueuing an existing item overwrites its timestamp.
- GetWork returns all items whose delay has expired and removes them from the queue.
- Uses an injectable clock for testability.

## Design Notes

- Thread-safe via mutex protection.
- Useful for scheduling pod syncs with backoff delays.
- Items are identified by pod UID, allowing deduplication of work.
- Clock abstraction allows deterministic testing.
