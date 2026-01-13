# Package: operationexecutor

## Purpose
Provides high-level execution of volume attach, detach, mount, and unmount operations with deduplication, using nestedpendingoperations to prevent concurrent operations on the same volume.

## Key Types/Structs
- `OperationExecutor` - Interface for executing volume operations
- `operationExecutor` - Implementation using NestedPendingOperations
- `VolumeToAttach/VolumeToMount` - Structs describing operation targets
- `ActualStateOfWorldMounterUpdater` - Interface for updating state after operations
- `OperationGenerator` - Generates operation functions for execution

## Key Functions
- `AttachVolume()` - Executes volume attach operation
- `DetachVolume()` - Executes volume detach operation
- `MountVolume()` - Executes volume mount (filesystem or block)
- `UnmountVolume()` - Executes volume unmount
- `UnmountDevice()` - Executes global device unmount
- `ExpandInUseVolume()` - Executes online volume expansion

## Design Patterns
- Wraps nestedpendingoperations for operation deduplication
- Separates operation generation from execution
- Updates actual state of world on operation completion
- Supports both filesystem and block volume operations
- Idempotent operations with state tracking
