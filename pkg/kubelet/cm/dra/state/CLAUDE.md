# Package state

Package state provides state types and checkpoint support for DRA (Dynamic Resource Allocation) claim tracking.

## Key Types

- `ClaimInfoState`: State for a single ResourceClaim
- `ClaimInfoStateList`: List of claim states for checkpointing
- `DriverState`: Per-driver device information within a claim
- `Device`: Individual device allocation details
- `DeviceHealth`: Health status of a device
- `DevicesHealthMap`: Map of driver -> device health states

## ClaimInfoState Fields

- `ClaimUID`: UID of the ResourceClaim
- `ClaimName`: Name of the ResourceClaim
- `Namespace`: Claim namespace
- `PodUIDs`: Set of pod UIDs referencing this claim
- `DriverState`: Map of driver name to DriverState

## Device Fields

- `PoolName`: Resource pool containing the device
- `DeviceName`: Device identifier within the pool
- `ShareID`: Optional UID for shared devices
- `RequestNames`: Request names this device satisfies
- `CDIDeviceIDs`: CDI device identifiers for container runtime

## Health Status Constants

- `DeviceHealthStatusHealthy`: Device is healthy
- `DeviceHealthStatusUnhealthy`: Device is unhealthy
- `DeviceHealthStatusUnknown`: Health status unknown

## Design Notes

- State is checkpointed to survive kubelet restarts
- Multiple pods can reference the same claim
- Health tracking enables pod status updates with device health
- DeepCopy generation for safe concurrent access
