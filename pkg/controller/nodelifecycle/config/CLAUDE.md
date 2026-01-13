# Package: config

Contains configuration types for the NodeLifecycle controller.

## Key Types

- **NodeLifecycleControllerConfiguration**: Configuration struct containing:
  - `NodeEvictionRate`: Pods deleted per second during normal zone health.
  - `SecondaryNodeEvictionRate`: Pods deleted per second when zone is unhealthy.
  - `NodeStartupGracePeriod`: Grace period for newly started nodes.
  - `NodeMonitorGracePeriod`: Time before marking unresponsive nodes as unhealthy.
  - `LargeClusterSizeThreshold`: Cluster size threshold affecting eviction behavior.
  - `UnhealthyZoneThreshold`: Percentage of NotReady nodes to trigger zone unhealthy state.

## Design Patterns

- Part of the componentconfig pattern used across Kubernetes controllers.
- Configuration is typically loaded from kube-controller-manager flags or config files.
