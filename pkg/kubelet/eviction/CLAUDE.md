# Package: eviction

Implements the kubelet's eviction manager that monitors node resources and evicts pods when thresholds are exceeded to maintain node stability.

## Key Types/Structs

- **managerImpl**: Main implementation of the eviction Manager interface. Monitors resource usage, tracks node conditions, and evicts pods when necessary.
- **Config**: Configuration for eviction including thresholds, grace periods, and kernel memcg notification settings.
- **Manager**: Interface for the eviction manager with methods to start monitoring and check pressure conditions.
- **DiskInfoProvider**: Interface to detect dedicated image/container filesystems.
- **ThresholdNotifier**: Manages cgroup notifiers for memory eviction thresholds.
- **CgroupNotifier**: Generates events from cgroup memory pressure events.

## Key Functions

- `NewManager()`: Creates a new eviction manager and returns it along with a PodAdmitHandler.
- `Start()`: Begins the control loop monitoring resources at specified intervals.
- `synchronize()`: Main control loop that checks thresholds, ranks pods, and triggers eviction.
- `IsUnderMemoryPressure() / IsUnderDiskPressure() / IsUnderPIDPressure()`: Query current node conditions.
- `localStorageEviction()`: Evicts pods exceeding local storage limits (emptyDir, ephemeral storage).
- `Admit()`: PodAdmitHandler that rejects pods when node is under resource pressure.

## Design Notes

- Monitors multiple signals: memory, nodefs, imagefs, containerfs, PIDs
- Supports both hard eviction (immediate) and soft eviction (with grace period)
- Uses cgroup memory notifications for faster response on Linux/Windows
- Pods ranked for eviction based on QoS class and resource usage
- Critical pods are never evicted; BestEffort pods evicted first under memory pressure
- Attempts node-level reclamation (image GC, container GC) before evicting pods
