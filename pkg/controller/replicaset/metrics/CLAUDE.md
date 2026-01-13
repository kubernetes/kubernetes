# Package: metrics

Provides Prometheus metrics for the ReplicaSet controller.

## Key Metrics

- **SortingDeletionAgeRatio**: Histogram tracking the ratio of the youngest to oldest pod ages when sorting pods for deletion. Higher values indicate more recent pods being deleted first.

## Design Patterns

- Uses histogram buckets optimized for ratio values (0 to 1).
- Helps diagnose pod deletion patterns and imbalances.
- Registered in the legacy Prometheus registry.
