# Package: metrics

## Purpose
Defines Prometheus metrics for volume binding operations in the scheduler. Tracks cache operations and scheduling stage errors.

## Key Metrics

### VolumeBindingRequestSchedulerBinderCache
Counter vector tracking volume binder cache operations:
- **Subsystem**: scheduler_volume
- **Name**: binder_cache_requests_total
- **Labels**: operation (e.g., "add", "delete", "assume")
- **Stability**: ALPHA

### VolumeSchedulingStageFailed
Counter vector tracking volume scheduling stage failures:
- **Subsystem**: scheduler_volume
- **Name**: scheduling_stage_error_total
- **Labels**: operation (e.g., "filter", "reserve", "prebind")
- **Stability**: ALPHA

## Key Functions

### RegisterVolumeSchedulingMetrics()
Registers all volume scheduling metrics with the legacy registry. Called during scheduler initialization.

## Metric Usage
```go
// Increment cache operation counter
metrics.VolumeBindingRequestSchedulerBinderCache.WithLabelValues("assume").Inc()

// Increment failure counter
metrics.VolumeSchedulingStageFailed.WithLabelValues("prebind").Inc()
```

## Design Pattern
- Uses component-base metrics framework
- Registered with legacy registry for backwards compatibility
- Alpha stability - metrics may change in future versions
- Labeled counters enable per-operation monitoring
