# Package: nestedpendingoperations

## Purpose
Manages concurrent volume operations, preventing duplicate operations on the same volume while allowing parallel operations on different pods for the same volume.

## Key Types/Structs
- `NestedPendingOperations` - Interface for managing pending operations
- `nestedPendingOperations` - Implementation tracking running operations
- `operation` - Tracks individual operation state and backoff

## Key Functions
- `NewNestedPendingOperations()` - Creates new operation manager
- `Run()` - Starts operation if not already pending for volume/pod/node key
- `Wait()` - Blocks until all operations complete
- `IsOperationPending()` - Checks if operation is in progress
- `IsOperationSafeToRetry()` - Checks if retry is safe (considering backoff)

## Design Patterns
- Composite key: volumeName + podName + nodeName for operation deduplication
- Exponential backoff on failures to prevent retry storms
- Allows parallel operations on same volume for different pods
- Prevents duplicate attach/mount operations causing race conditions
- Thread-safe with mutex protection
