# Package: metrics

Provides Prometheus metrics for the StatefulSet controller.

## Key Metrics

- **MaxUnavailable**: Gauge tracking the configured `.spec.updateStrategy.rollingUpdate.maxUnavailable` value per StatefulSet. Labels: statefulset_namespace, statefulset_name, pod_management_policy.

- **UnavailableReplicas**: Gauge tracking the current number of unavailable pods in a StatefulSet (missing or not ready for minReadySeconds). Labels: statefulset_namespace, statefulset_name, pod_management_policy.

## Design Patterns

- Both metrics are ALPHA stability level.
- Useful for monitoring rolling update progress and availability.
- Labeled by pod management policy (OrderedReady vs Parallel).
- Supports alerting when unavailable exceeds max unavailable.
