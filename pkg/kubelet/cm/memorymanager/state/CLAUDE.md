# Package state

Package state provides state management for memory manager, tracking NUMA memory assignments per container with checkpoint persistence.

## Key Types

- `State`: Combined interface for reading and writing memory state
- `Reader`: Read-only interface for memory assignments
- `writer`: Write interface for memory assignments (unexported)
- `MemoryTable`: Memory accounting for a NUMA node
- `NUMANodeState`: State for a single NUMA node
- `NUMANodeMap`: Map from NUMA node ID to state
- `Block`: Memory allocation with NUMA affinity
- `ContainerMemoryAssignments`: Map from podUID -> containerName -> Blocks

## MemoryTable Fields

- `TotalMemSize`: Total memory on NUMA node
- `SystemReserved`: Memory reserved for system
- `Allocatable`: Memory available for allocation
- `Reserved`: Memory reserved for containers
- `Free`: Remaining free memory

## Block Fields

- `NUMAAffinity`: List of NUMA node IDs
- `Type`: Resource type (memory or hugepages-*)
- `Size`: Amount of memory in bytes

## Reader Methods

- `GetMachineState()`: Returns NUMA node memory map
- `GetMemoryBlocks(podUID, containerName)`: Returns container's memory blocks
- `GetMemoryAssignments()`: Returns all container assignments

## Implementations

- `stateMemory`: In-memory state (volatile)
- `stateCheckpoint`: File-backed with checkpoint persistence

## Design Notes

- Supports multiple NUMA node affinity for cross-NUMA allocation
- Clone methods for safe state copying
- Checksum verification for data integrity
