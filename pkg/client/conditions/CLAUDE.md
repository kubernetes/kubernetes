# Package: conditions

## Purpose
Provides condition functions for use with watch.Event processing, particularly for waiting on pod state changes.

## Key Variables
- `ErrPodCompleted`: Error returned when a pod has already run to completion

## Key Functions
- `PodRunning(event watch.Event) (bool, error)`: Returns true if pod is running, false if not yet running, ErrPodCompleted if finished, or error if deleted
- `PodCompleted(event watch.Event) (bool, error)`: Returns true if pod has run to completion (Failed or Succeeded phase)

## Usage Pattern
These functions are designed to be used with watch.Until or similar constructs:
```go
ctx, cancel := context.WithTimeout(ctx, timeout)
event, err := watch.Until(ctx, watchList, conditions.PodRunning)
```

## Design Notes
- Handles watch.Deleted events by returning NotFound error
- Distinguishes between "not yet ready" (returns false, nil) and terminal states
- Used by kubectl wait and other components that need to wait for pod states
