# Package: metrics

Defines interfaces and types for fetching metrics used by the HPA controller.

## Key Types

- **PodMetric**: Contains metric value, timestamp, and window for a single pod.
- **PodMetricsInfo**: Map from pod names to their PodMetric values.
- **MetricsClient**: Interface for fetching various metric types.

## Key Interface Methods (MetricsClient)

- **GetResourceMetric**: Fetches resource metrics (CPU, memory) for pods matching a selector.
- **GetRawMetric**: Fetches custom pod metrics by name.
- **GetObjectMetric**: Fetches metrics from a specific Kubernetes object.
- **GetExternalMetric**: Fetches metrics from external sources.

## Design Patterns

- Metric values are stored as milli-values for precision.
- Missing metrics don't cause errors; callers filter based on pod status.
- Container parameter allows fetching metrics for specific containers or sum of all.
- Supports metric selectors for filtering external and custom metrics.
