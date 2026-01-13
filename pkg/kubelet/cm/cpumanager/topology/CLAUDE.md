# Package topology

Package topology provides CPU topology discovery and representation for NUMA-aware CPU allocation.

## Key Types

- `CPUTopology`: Complete CPU topology (CPUs, cores, sockets, NUMA nodes, uncore caches)
- `CPUDetails`: Map from CPU ID to CPUInfo (core, socket, NUMA, uncore cache IDs)
- `CPUInfo`: Topology information for a single logical CPU
- `NUMANodeInfo`: Map from NUMA node ID to associated CPU set

## Key Functions

- `Discover(logger, machineInfo)`: Discovers CPU topology from cadvisor machine info
- `CPUsPerCore/CPUsPerSocket/CPUsPerUncore`: Compute CPU counts per topology level
- `CheckAlignment(cpuset)`: Checks if CPUs are aligned at topology boundaries

## CPUDetails Query Methods

- `CPUs()`: All logical CPU IDs
- `Cores()/Sockets()/NUMANodes()/UncoreCaches()`: IDs at each topology level
- `CPUsInCores/CPUsInSockets/CPUsInNUMANodes/CPUsInUncoreCaches`: CPUs in specified topology units
- `CoresInNUMANodes/CoresInSockets`: Cores in specified units
- `SocketsInNUMANodes/NUMANodesInSockets`: Cross-level queries
- `KeepOnly(cpuset)`: Filter to specified CPUs

## Topology Levels (highest to lowest)

1. Socket (physical CPU package)
2. NUMA Node (memory domain)
3. Uncore Cache (L3 cache domain)
4. Core (physical core)
5. CPU (logical CPU / hardware thread)

## Design Notes

- Core IDs normalized to lowest thread ID for platform uniqueness
- Uncore cache support for Intel L3 cache topology
- Falls back to socket alignment if uncore cache info unavailable
