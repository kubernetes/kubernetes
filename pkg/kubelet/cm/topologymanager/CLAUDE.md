# Package topologymanager

Package topologymanager coordinates NUMA-aware resource allocation across multiple hint providers (CPU, memory, device managers) to achieve optimal resource locality.

## Key Types

- `Manager`: Interface for topology management and pod admission
- `HintProvider`: Interface for components providing topology hints
- `Store`: Interface for retrieving pod affinity decisions
- `TopologyHint`: NUMA affinity with preferred flag
- `TopologyAffinityError`: Error when topology alignment fails

## Manager Interface

- `AddHintProvider`: Registers a hint provider (CPU, memory, device manager)
- `AddContainer/RemoveContainer`: Container lifecycle tracking
- `Admit`: Pod admission handler (implements PodAdmitHandler)
- `GetAffinity`: Returns stored topology hint for a container
- `GetPolicy`: Returns the active policy

## Policies

- `none`: No topology awareness
- `best-effort`: Prefer aligned allocation, allow misalignment
- `restricted`: Require alignment, reject on failure
- `single-numa-node`: Require all resources on single NUMA node

## Scopes

- `container`: Per-container topology alignment
- `pod`: Per-pod topology alignment

## Constants

- `defaultMaxAllowableNUMANodes`: 8 (state explosion limit)
- `ErrorTopologyAffinity`: Error type for topology failures

## Design Notes

- Collects hints from all providers, merges to find best alignment
- Hint preference indicates whether allocation can satisfy request
- BitMask represents NUMA node affinity
- Metrics track admission latency and success/failure
