# Package: operationexecutor

Executes plugin register/unregister operations with concurrency control.

## Key Interfaces

- **OperationExecutor**: Executes register/unregister operations
  - `RegisterPlugin()`: Registers a plugin and updates actual state
  - `UnregisterPlugin()`: Unregisters a plugin and updates actual state

- **ActualStateOfWorldUpdater**: Updates actual state cache
  - `AddPlugin()`: Adds plugin to cache
  - `RemovePlugin()`: Removes plugin from cache

- **OperationGenerator**: Generates the actual registration/unregistration functions

## Key Types

- **operationExecutor**: Implementation using GoRoutineMap for safe concurrent operations.

## Key Functions

- `NewOperationExecutor(generator)`: Creates executor with exponential backoff on errors.
- `IsOperationPending(socketPath)`: Checks if an operation is in progress for a socket.

## Design Notes

- Uses goroutinemap to prevent concurrent operations on the same socket path
- Operations are idempotent (RegisterPlugin succeeds if already registered)
- Errors are logged and goroutine terminates without updating state
- Successful operations update actualStateOfWorld cache
- Exponential backoff enabled for error recovery
