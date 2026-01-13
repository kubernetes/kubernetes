# Package: kuberuntime

Implementation of kubecontainer.Runtime using the CRI (Container Runtime Interface).

## Key Types/Structs

- **kubeGenericRuntimeManager**: Main runtime manager implementing Runtime, StreamingRuntime, and CommandRunner interfaces. Manages pod sandboxes, containers, images, and logs through CRI.
- **KubeGenericRuntime**: Combined interface for runtime, streaming, and command execution.
- **podStateProvider**: Interface for checking pod termination and removal status.

## Key Components

- Pod sandbox management (create, stop, remove)
- Container lifecycle (create, start, stop, kill, remove)
- Image management via CRI ImageManagerService
- Container log management
- Garbage collection for containers
- RuntimeClass integration for multi-runtime support
- Resource allocation tracking (actuatedState)

## Configuration Options

- `cpuCFSQuota`: Enforce CPU limits with CFS quota
- `cpuCFSQuotaPeriod`: CFS quota period (default 100ms)
- `seccompDefault`: Use RuntimeDefault seccomp profile
- `memorySwapBehavior`: Swap usage configuration
- `memoryThrottlingFactor`: Memory QoS throttling factor

## Key Integrations

- CRI RuntimeService and ImageManagerService for container operations
- ContainerManager for cgroup management
- ProbeResults managers (liveness, readiness, startup)
- LogManager for container log rotation
- RuntimeClassManager for runtime handler selection

## Design Notes

- Uses OpenTelemetry tracing for observability
- Caches runtime version with configurable TTL
- Reduces log spam with LogReduction
- Supports both single and multi-container OOM kill modes
