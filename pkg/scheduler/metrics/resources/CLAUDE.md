# Package: resources

## Purpose
Provides resource-specific metrics for the scheduler, particularly around pod resource requests and node capacity tracking.

## Key Types
- `podResourceCollector` - Prometheus collector for pod resource metrics
- `resourceLifecycleDescriptors` - Describes resource metric lifecycle

## Key Functions
- `NewPodResourcesMetricsCollector()` - Creates collector for pod resource metrics
- `Describe()` - Returns metric descriptors for Prometheus
- `Collect()` - Gathers current resource metrics from the cluster

## Key Metrics
- Pod resource requests (CPU, memory, etc.) by namespace and priority
- Scheduled vs unscheduled pod resource breakdowns
- Resource utilization across the scheduling queue

## Design Patterns
- Implements Prometheus Collector interface for custom metrics
- Aggregates resources across pods for efficient collection
- Separates scheduled and pending pod resource accounting
