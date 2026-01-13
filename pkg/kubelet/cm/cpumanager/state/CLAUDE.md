# Package state

Package state provides state management for CPU manager, tracking CPU assignments per container with checkpoint persistence.

## Key Types

- `State`: Combined interface for reading and writing CPU state
- `Reader`: Read-only interface for CPU assignments
- `writer`: Write interface for CPU assignments (unexported)
- `ContainerCPUAssignments`: Map from podUID -> containerName -> CPUSet

## Implementations

- `stateMemory`: In-memory state storage (no persistence)
- `stateCheckpoint`: File-backed state with checkpoint persistence

## Reader Interface Methods

- `GetCPUSet(podUID, containerName)`: Returns assigned CPUSet for a container
- `GetDefaultCPUSet()`: Returns the shared/default CPU pool
- `GetCPUSetOrDefault(podUID, containerName)`: Returns assigned or default CPUSet
- `GetCPUAssignments()`: Returns all container CPU assignments

## Writer Interface Methods

- `SetCPUSet(podUID, containerName, cpuset)`: Assigns CPUs to a container
- `SetDefaultCPUSet(cpuset)`: Sets the shared CPU pool
- `SetCPUAssignments(assignments)`: Bulk update assignments
- `Delete(podUID, containerName)`: Removes container's CPU assignment
- `ClearState()`: Resets all state

## Design Notes

- Checkpoint state uses JSON serialization with checksum verification
- Memory state useful for testing or volatile configurations
- State is restored from checkpoint on kubelet restart
