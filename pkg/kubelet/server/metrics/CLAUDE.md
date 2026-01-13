# Package: metrics

## Purpose
The `metrics` package defines Prometheus metrics for monitoring the Kubelet HTTP server's performance and behavior.

## Key Metrics

- **HTTPRequests**: Counter tracking total HTTP requests received, labeled by method, path, server_type (readonly/readwrite), and long_running status.
- **HTTPRequestsDuration**: Histogram measuring request duration in seconds with the same labels.
- **HTTPInflightRequests**: Gauge tracking currently in-flight HTTP requests.
- **VolumeStatCalDuration**: Histogram measuring duration of volume statistics calculations, labeled by metric_source.

## Key Functions

- **Register**: Registers all metrics with the legacy registry (called once via sync.Once).
- **SinceInSeconds**: Helper to calculate elapsed time since a start time in seconds.
- **CollectVolumeStatCalDuration**: Records volume stat calculation duration for a given metric source.

## Design Notes

- All metrics use the "kubelet" subsystem prefix.
- Metrics are registered with ALPHA stability level.
- The server_type label distinguishes between read-only and read-write server instances.
- Long-running requests (exec, attach, portforward, debug) are tracked separately.
