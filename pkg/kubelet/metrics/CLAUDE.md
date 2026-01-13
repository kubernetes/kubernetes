# Package: metrics

Defines and registers Prometheus metrics for kubelet monitoring.

## Metric Categories

### Pod/Container Startup
- `pod_start_duration_seconds`, `pod_start_sli_duration_seconds`, `pod_start_total_duration_seconds`
- `started_pods_total`, `started_containers_total` (with error variants)
- `first_network_pod_start_sli_duration_seconds`

### Node Startup
- `node_startup_duration_seconds`, `node_startup_pre_kubelet_duration_seconds`
- `node_startup_registration_duration_seconds`

### Runtime Operations
- `runtime_operations_total`, `runtime_operations_duration_seconds`, `runtime_operations_errors_total`
- `run_podsandbox_duration_seconds`, `run_podsandbox_errors_total`

### PLEG (Pod Lifecycle Event Generator)
- `pleg_relist_duration_seconds`, `pleg_relist_interval_seconds`, `pleg_last_seen_seconds`
- `evented_pleg_connection_*` metrics

### Resource Management
- `evictions`, `preemptions`
- `cpu_manager_*`, `memory_manager_*`, `topology_manager_*` metrics
- Pod resize metrics: `pod_resize_duration_milliseconds`, `pod_pending_resizes`

### Image/Volume
- `image_pull_duration_seconds`, `image_garbage_collected_total`
- `volume_stats_*` metrics (capacity, available, used bytes/inodes)

### DRA (Dynamic Resource Allocation)
- `dra_operations_duration_seconds`, `dra_grpc_operations_duration_seconds`

## Key Functions

- `Register()`: Registers all kubelet metrics with the legacy registry
- `GetImageSizeBucket()`: Returns human-readable size bucket for image pull metrics
- `SinceInSeconds()`: Helper for duration calculations

## Design Notes

- Uses component-base metrics library for Prometheus integration
- Metrics organized under "kubelet" subsystem
- Some metrics gated by feature flags
