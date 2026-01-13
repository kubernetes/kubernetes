# Package memorymanager

Package memorymanager provides NUMA-aware memory allocation for containers, ensuring memory locality for performance-sensitive workloads.

## Key Types

- `Manager`: Interface for memory management operations
- `manager`: Concrete implementation tracking memory assignments
- `Policy`: Interface for memory allocation policies

## Manager Interface Methods

- `Start`: Initializes the manager with active pods and starts state tracking
- `Allocate`: Pre-allocates memory during pod admission
- `AddContainer/RemoveContainer`: Manages container lifecycle in memory tracking
- `GetTopologyHints`: Provides NUMA hints for topology-aware scheduling
- `GetMemoryNUMANodes`: Returns NUMA nodes used for container memory
- `GetAllocatableMemory`: Returns allocatable memory per NUMA node
- `GetMemory`: Returns memory blocks allocated to a container

## Policies

- `policyTypeNone`: No NUMA memory management
- `PolicyTypeStatic`: Strict NUMA memory allocation (Linux only)

## Constants

- `memoryManagerStateFileName`: "memory_manager_state"

## State Tracking

- Memory blocks with NUMA affinity
- Per-NUMA node memory tables (total, reserved, free, allocatable)
- Container to memory block assignments

## Design Notes

- Not available on Windows (static policy)
- Integrates with topology manager for multi-resource alignment
- Checkpoints state for persistence across restarts
- Supports both regular memory and hugepages allocation
