# Package: types

## Purpose
Defines common types used across volume packages, including operation types, error types, and unique identifiers.

## Key Types/Structs
- `UniquePodName` - Type for uniquely identifying pods (types.UID)
- `UniquePVCName` - Type for uniquely identifying PVCs
- `GeneratedOperations` - Container for operation function and callbacks
- `OperationContext` - Result context from operation execution
- `CompleteFuncParam` - Parameters passed to completion callback

## Error Types
- `TransientOperationFailure` - Retryable operation failure
- `UncertainProgressError` - Operation may be in progress (uncertain state)
- `FailedPrecondition` - CSI precondition failure
- `InfeasibleError` - Operation not possible in current state
- `OperationNotSupported` - Operation not supported by driver

## Key Functions
- `GeneratedOperations.Run()` - Executes operation with callbacks
- `IsOperationFinishedError()` - Checks if error is final (not retryable)
- `IsFilesystemMismatchError()` - Checks for FS mismatch on mount
- Error constructors and type checkers for each error type

## Design Patterns
- Semantic error types for proper error handling and retry logic
- Operation abstraction with completion callbacks
- Supports CSI migration tracking via Migrated flag
