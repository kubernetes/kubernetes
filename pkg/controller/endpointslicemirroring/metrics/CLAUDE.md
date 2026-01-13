# Package: metrics

Prometheus metrics for the EndpointSlice mirroring controller.

## Key Metrics

- `endpoints_added_per_sync`: Histogram of endpoints added per sync
- `endpoints_updated_per_sync`: Histogram of endpoints updated per sync
- `endpoints_removed_per_sync`: Histogram of endpoints removed per sync
- `addresses_skipped_per_sync`: Histogram of addresses skipped (invalid or over limit)
- `endpoints_sync_duration`: Histogram of sync duration in seconds
- `endpoints_desired`: Gauge of total desired endpoints
- `num_endpoint_slices`: Gauge of actual EndpointSlice count
- `desired_endpoint_slices`: Gauge of ideal EndpointSlice count with perfect allocation
- `changes`: Counter of EndpointSlice changes by operation type

## Key Functions

- `RegisterMetrics()`: Registers all metrics with the legacy registry (called once via sync.Once)

## Purpose

Provides observability into the EndpointSlice mirroring controller's operation, tracking sync performance, endpoint counts, and EndpointSlice efficiency.

## Design Notes

- All metrics use the `endpoint_slice_mirroring_controller` subsystem
- Histograms use exponential buckets for wide range coverage
