# Package: metrics

## Purpose
Defines and registers Prometheus metrics for the TTL After Finished controller to track job deletion performance.

## Key Constants

- **TTLAfterFinishedSubsystem**: Subsystem name "ttl_after_finished_controller" used for metric naming.

## Key Variables

- **JobDeletionDurationSeconds**: Histogram metric tracking time elapsed between when a Job became eligible for deletion and when it was actually deleted.
  - Buckets: Exponential starting at 0.1s, factor of 2, 14 buckets (up to ~27 minutes)
  - Stability level: ALPHA

## Key Functions

- **Register()**: Thread-safe registration of metrics with the legacy registry (called once via sync.Once).

## Design Notes

- Uses `k8s.io/component-base/metrics` for Kubernetes-compatible metric definitions.
- Metrics are registered with the legacy registry for backward compatibility.
- The histogram helps identify delays in job cleanup processing.
