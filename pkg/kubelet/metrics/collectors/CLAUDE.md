# Package: collectors

Custom Prometheus metric collectors for kubelet resource and stats metrics.

## Key Collectors

### resourceMetricsCollector
Exports resource usage metrics for nodes, pods, and containers:

**Node Metrics:**
- `node_cpu_usage_seconds_total`: Cumulative CPU time (STABLE)
- `node_memory_working_set_bytes`: Current memory working set (STABLE)
- `node_swap_usage_bytes`: Current swap usage (ALPHA)

**Container Metrics:**
- `container_cpu_usage_seconds_total`: Cumulative CPU time (STABLE)
- `container_memory_working_set_bytes`: Memory working set (STABLE)
- `container_swap_usage_bytes`, `container_swap_limit_bytes`: Swap metrics (ALPHA)
- `container_start_time_seconds`: Container start time (STABLE)

**Pod Metrics:**
- `pod_cpu_usage_seconds_total`, `pod_memory_working_set_bytes` (STABLE)
- `pod_swap_usage_bytes` (ALPHA)

### Other Collectors
- **logMetricsCollector**: Container log file metrics
- **volumeStatsCollector**: Volume capacity/usage metrics
- **criMetricsCollector**: CRI-level metrics
- **podcertificateMetricsCollector**: Pod certificate state metrics

## Key Functions

- `NewResourceMetricsCollector(provider)`: Creates resource metrics collector
- `CollectWithStability()`: Custom collection for active containers only

## Design Notes

- Uses custom collectors instead of gauges to avoid metric leaks for removed containers
- Implements metrics.StableCollector interface
- Labeled by container, pod, namespace where applicable
- Metrics include timestamps for time-series accuracy
