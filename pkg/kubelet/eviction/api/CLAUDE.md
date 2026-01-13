# Package: api

Defines the API types for kubelet eviction thresholds and signals.

## Key Types/Structs

- **Signal**: String type representing a resource signal that can trigger eviction (e.g., "memory.available", "nodefs.available").
- **ThresholdOperator**: Operator for threshold comparison (currently only "LessThan" supported).
- **ThresholdValue**: Holds either an absolute Quantity or a Percentage for threshold comparison.
- **Threshold**: Complete threshold definition with Signal, Operator, Value, GracePeriod, and MinReclaim.

## Signal Constants

- `SignalMemoryAvailable`: Available memory (capacity - workingSet)
- `SignalNodeFsAvailable`: Storage available on kubelet's filesystem
- `SignalNodeFsInodesFree`: Inodes available on kubelet's filesystem
- `SignalImageFsAvailable`: Storage for container image layers
- `SignalImageFsInodesFree`: Inodes for image layers
- `SignalContainerFsAvailable`: Storage for container writable layers
- `SignalContainerFsInodesFree`: Inodes for container writable layers
- `SignalAllocatableMemoryAvailable`: Memory available for pod allocation
- `SignalPIDAvailable`: PIDs available for pod allocation

## Key Functions

- `GetThresholdQuantity()`: Converts ThresholdValue to absolute Quantity given capacity.

## Design Notes

- All signals use "LessThan" operator (evict when resource falls below threshold)
- Thresholds can be specified as absolute values or percentages
- OpForSignal map provides operator lookup for each signal type
