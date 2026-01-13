# Package cpumanager

Package cpumanager provides CPU affinity management for containers, enabling exclusive CPU allocation for guaranteed QoS pods.

## Key Types

- `Manager`: Interface for CPU management operations (allocation, removal, topology hints)
- `Policy`: Interface for CPU assignment logic (none, static policies)
- `manager`: Concrete implementation tracking CPU assignments and reconciling state

## Key Functions

- `NewManager`: Creates a CPU manager with specified policy and configuration
- `Start`: Initializes the manager with active pods and starts reconciliation loop
- `Allocate`: Assigns CPUs to a container based on policy
- `AddContainer/RemoveContainer`: Manages container lifecycle in CPU tracking
- `GetExclusiveCPUs`: Returns exclusively allocated CPUs for a container
- `GetTopologyHints`: Provides NUMA hints for topology-aware scheduling

## Policies

- `PolicyNone`: No CPU pinning, all containers share the shared pool
- `PolicyStatic`: Exclusive CPU allocation for guaranteed QoS containers with integer CPU requests

## State Files

- `cpu_manager_state`: Checkpoint file for CPU assignments (persisted across restarts)

## Platform Support

- Linux: Full support with cgroup CPU affinity
- Windows: Policy support but different cgroup integration
- Others: Stub implementation

## Design Notes

- Reconciliation loop periodically syncs container CPU affinity with desired state
- Integrates with topology manager for NUMA-aware allocation
- Reserved CPUs (system + kube reserved) excluded from allocation pool
- Fractional CPU requests use shared pool, integer requests can get exclusive CPUs
