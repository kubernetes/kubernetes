# Package: metrics

Prometheus metrics for the device taint eviction controller.

## Key Metrics

- `device_taint_eviction_controller_pod_deletions_total`: Counter tracking total number of pods deleted
- `device_taint_eviction_controller_pod_deletion_duration_seconds`: Histogram tracking latency between taint effect activation and pod deletion (buckets from 5ms to 4 minutes)

## Key Types

- `Metrics`: Contains all metrics supported by the controller, implements `metrics.Gatherer`

## Key Functions

- `Register()`: Registers metrics with the legacy registry (called once via sync.Once)
- `New()`: Creates new metric instances, useful for parallel testing with custom buckets

## Purpose

Provides observability into the device taint eviction controller's operation, tracking how many pods are evicted and how quickly the controller responds to taint changes.
