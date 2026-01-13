# Package: stats

## Purpose
The `stats` package collects and provides container and pod statistics for the Kubelet. It implements stats providers using both CRI (Container Runtime Interface) and cAdvisor as data sources.

## Key Types/Structs

- **Provider**: Interface for getting stats from CRI, cAdvisor, and the Kubelet including pod stats, cgroup stats, filesystem stats, and node info.
- **criStatsProvider**: Collects container stats from CRI runtime. Includes CPU usage caching to calculate nano core usage.
- **cadvisorStatsProvider**: Collects container stats from cAdvisor for runtimes that don't provide CRI stats.
- **cpuUsageRecord**: Cache entry holding CPU usage stats and calculated nano core usage for a container.
- **HostStatsProvider**: Interface for getting host-level stats like pod container log stats and ephemeral storage.

## Key Functions

- **NewCRIStatsProvider**: Creates a stats provider that gets stats from CRI.
- **newCadvisorStatsProvider**: Creates a stats provider that gets stats from cAdvisor.
- **ListPodStats**: Returns stats for all pod-managed containers.
- **ListPodStatsAndUpdateCPUNanoCoreUsage**: Returns pod stats while updating CPU nano core usage cache.
- **ListPodCPUAndMemoryStats**: Returns only CPU and memory stats for all pods (used by eviction manager).
- **ImageFsStats**: Returns filesystem stats for image storage.
- **getCadvisorContainerInfo**: Fetches container info from cAdvisor recursively.

## Design Notes

- CRI stats provider maintains a CPU usage cache to compute UsageNanoCores when the runtime doesn't provide it directly.
- Supports fallback from CRI to cAdvisor when CRI doesn't implement certain stats APIs.
- Filters terminated containers to avoid duplicate entries in stats.
- Pod-level stats are aggregated from container stats or read from pod cgroup.
- Supports PSI (Pressure Stall Information) stats when KubeletPSI feature is enabled.
- Host stats provider handles platform-specific (Linux/Windows) storage stats.
