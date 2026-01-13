# Package: goroutinemap

## Purpose
Manages named goroutines, preventing duplicate operations and optionally implementing exponential backoff on errors.

## Key Types
- `GoRoutineMap` - Interface for running and tracking named goroutines
- `goRoutineMap` - Implementation with operation tracking and backoff support
- `operation` - Tracks pending state and backoff info for each operation

## Key Functions
- `NewGoRoutineMap()` - Creates a new goroutine map with optional backoff
- `Run()` - Starts a named operation if not already running
- `Wait()` - Blocks until all operations complete
- `WaitForCompletion()` - Blocks until operations complete or fail (not pending)
- `IsOperationPending()` - Checks if an operation is currently running

## Error Types
- `AlreadyExistsError` - Operation with same name is already running
- `ExponentialBackoffError` - Operation failed recently and is in backoff period

## Design Patterns
- Prevents duplicate concurrent operations by name
- Optional exponential backoff (500ms to ~2min) on errors
- Condition variable for efficient waiting
- Panic recovery in goroutines
- Used by volume attach/detach operations
