# Package containermap

Package containermap provides a bidirectional mapping between container IDs and pod/container references.

## Key Types

- `ContainerMap`: Map from containerID to (podUID, containerName) pairs
- `cmItem`: Internal struct holding podUID and containerName

## Key Functions

- `NewContainerMap()`: Creates a new empty ContainerMap
- `Clone()`: Creates a deep copy of the ContainerMap
- `Add(podUID, containerName, containerID)`: Adds a mapping
- `RemoveByContainerID(containerID)`: Removes mapping by container ID
- `RemoveByContainerRef(podUID, containerName)`: Removes mapping by pod/container reference
- `GetContainerID(podUID, containerName)`: Looks up container ID from pod/container ref
- `GetContainerRef(containerID)`: Looks up pod/container ref from container ID
- `Visit(visitor)`: Iterates over all entries with a callback function

## Design Notes

- Used by CPU manager, memory manager, and device manager to track container allocations
- Efficient lookup by container ID (O(1))
- Lookup by pod/container reference requires iteration (O(n))
- Thread-safety must be provided by the caller
