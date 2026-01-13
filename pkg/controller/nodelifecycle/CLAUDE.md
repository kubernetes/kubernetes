# Package: nodelifecycle

Manages node lifecycle by monitoring node health, applying taints, and coordinating pod eviction from unhealthy nodes.

## Key Types

- **Controller**: Main controller that monitors node health, manages taints, and triggers evictions.
- **ZoneState**: Represents zone health states (Initial, Normal, PartialDisruption, FullDisruption).
- **nodeHealthData**: Tracks node health information including probe timestamps and ready transitions.
- **nodeHealthMap**: Thread-safe map storing health data for all nodes.

## Key Functions

- **NewNodeLifecycleController**: Creates a new controller with configurable grace periods and eviction rates.
- **Run**: Starts the controller, launching workers for node updates, pod updates, and health monitoring.
- **monitorNodeHealth**: Periodically checks node health and updates conditions to Unknown if stale.
- **processTaintBaseEviction**: Manages NoExecute taints based on node ready conditions.
- **handleDisruption**: Adjusts eviction rate based on zone health (stops eviction in full disruption).
- **ComputeZoneState**: Determines zone state from node ready conditions.

## Key Taints

- **UnreachableTaintTemplate**: Applied when node condition is Unknown.
- **NotReadyTaintTemplate**: Applied when node condition is False.

## Design Patterns

- Uses rate-limited timed queues per zone for controlled eviction.
- Implements zone-aware disruption handling to prevent cascade failures.
- Reconciles legacy beta labels with stable labels (OS, Arch).
- Optionally runs TaintEvictionController inline (controlled by feature gate).
- Exposes Prometheus metrics for zone health, size, and evictions.
