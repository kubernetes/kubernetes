# Package: metrics

## Purpose
Provides Prometheus metrics for monitoring the Kubernetes scheduler, including scheduling latency, queue depth, and plugin execution times.

## Key Types
- `MetricRecorder` - Interface for recording scheduling metrics
- `PendingPodsRecorder` - Records pending pod counts by queue type
- `metricRecorderImpl` - Default implementation of MetricRecorder

## Key Functions
- `Register()` - Registers all scheduler metrics with Prometheus
- `SinceInSeconds()` - Calculates duration since a given time
- `NewMetricRecorder()` - Creates a new metric recorder instance
- `RecordSchedulingLatency()` - Records end-to-end scheduling latency

## Key Metrics
- `scheduler_scheduling_attempt_duration_seconds` - Scheduling attempt latency histogram
- `scheduler_pending_pods` - Number of pending pods by queue
- `scheduler_plugin_execution_duration_seconds` - Plugin execution time
- `scheduler_framework_extension_point_duration_seconds` - Extension point latency

## Design Patterns
- Uses Prometheus client library for metric registration
- Supports labeled metrics for filtering by profile, result, and extension point
- Thread-safe metric recording
