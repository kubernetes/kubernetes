# Package checkpoint

Package checkpoint provides checkpoint data structures for persisting device plugin allocations across kubelet restarts.

## Key Types

- `DeviceManagerCheckpoint`: Interface extending checkpointmanager.Checkpoint with GetData()
- `Data`: Checkpoint data with checksum verification
- `PodDevicesEntry`: Pod to device allocation mapping
- `DevicesPerNUMA`: Device IDs per NUMA node (map[int64][]string)
- `checkpointData`: Internal struct with pod entries and registered devices

## Key Functions

- `New(devEntries, devices)`: Creates a new checkpoint (v2 format)
- `NewDevicesPerNUMA()`: Creates an empty NUMA device map
- `Devices()`: Returns all device IDs across NUMA nodes as a set

## PodDevicesEntry Fields

- `PodUID`: Pod identifier
- `ContainerName`: Container within the pod
- `ResourceName`: Extended resource name (e.g., nvidia.com/gpu)
- `DeviceIDs`: Device IDs per NUMA node
- `AllocResp`: Serialized allocation response from plugin

## Checkpoint Interface

- `MarshalCheckpoint()`: Serializes with checksum
- `UnmarshalCheckpoint(blob)`: Deserializes JSON
- `VerifyChecksum()`: Validates integrity
- `GetData()`: Returns entries and registered devices

## Design Notes

- NUMA-aware device tracking for topology-aware allocation
- Stores allocation responses for container restart scenarios
- Checksum prevents corruption detection
