# Package cm (Container Manager)

Package cm provides the kubelet's container manager which handles cgroups, resource allocation, and device/CPU/memory management for pods and containers.

## Key Types

- `ContainerManager`: Main interface for container resource management
- `NodeConfig`: Node-level configuration for cgroups, QoS, CPU/memory/topology managers
- `NodeAllocatableConfig`: Configuration for reserved resources and enforcement
- `Status`: Container manager status including soft requirement errors

## ContainerManager Interface

Key responsibilities:
- Cgroup management (system, kubelet, pod cgroups)
- QoS class container management (Guaranteed, Burstable, BestEffort)
- Node allocatable resource calculation and enforcement
- Device plugin integration
- CPU manager for exclusive CPU allocation
- Memory manager for NUMA-aware memory allocation
- Topology manager for resource alignment
- Dynamic Resource Allocation (DRA) support

## Platform Support

- `container_manager_linux.go`: Full cgroup v1/v2 support
- `container_manager_windows.go`: Windows-specific implementation
- `container_manager_unsupported.go`: Stub for unsupported platforms
- `container_manager_stub.go`: Minimal implementation for testing

## Sub-packages

- `cpumanager`: CPU pinning and exclusive CPU allocation
- `memorymanager`: NUMA-aware memory allocation
- `devicemanager`: Device plugin framework
- `topologymanager`: Cross-resource topology alignment
- `dra`: Dynamic Resource Allocation support

## Design Notes

- Implements podresources.CPUsProvider, DevicesProvider, MemoryProvider, DynamicResourcesProvider
- Cgroup v1 is deprecated, v2 is recommended
- Supports percentage-based QoS reservations for memory
