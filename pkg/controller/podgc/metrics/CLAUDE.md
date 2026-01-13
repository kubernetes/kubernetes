# Package: metrics

Provides Prometheus metrics for the Pod GC controller.

## Key Metrics

- **DeletingPodsTotal**: Counter of pods deleted, labeled by namespace and reason.
- **DeletingPodsErrorTotal**: Counter of deletion errors, labeled by namespace and reason.

## Reason Labels

- `PodGCReasonTerminated`: Pod exceeded terminated pod threshold.
- `PodGCReasonOrphaned`: Pod assigned to non-existent node.
- `PodGCReasonTerminatingOutOfService`: Terminating pod on out-of-service node.
- `PodGCReasonTerminatingUnscheduled`: Unscheduled terminating pod.

## Design Patterns

- Metrics labeled by namespace for per-namespace observability.
- Separate error counter helps identify deletion failures.
- Reason labels distinguish between different GC triggers.
