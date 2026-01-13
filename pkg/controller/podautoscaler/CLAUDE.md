# Package: podautoscaler

Implements the Horizontal Pod Autoscaler (HPA) controller that automatically scales workloads based on metrics.

## Key Types

- **HorizontalController**: Main controller that reconciles HPA objects with target scale resources.
- **ReplicaCalculator**: Computes desired replica counts from various metric types.
- **timestampedRecommendation**: Stores replica recommendations with timestamps for stabilization.
- **timestampedScaleEvent**: Tracks scale events for rate limiting.
- **NormalizationArg**: Arguments for replica count normalization with behaviors.

## Key Functions

- **NewHorizontalController**: Creates controller with metrics client, informers, and configuration.
- **Run**: Starts workers that process HPA reconciliation.
- **reconcileAutoscaler**: Main reconciliation logic - fetches metrics, computes replicas, updates scale.
- **computeReplicasForMetrics**: Calculates desired replicas across all metric specifications.
- **normalizeDesiredReplicas**: Applies min/max bounds and scale-up limits.
- **normalizeDesiredReplicasWithBehaviors**: Applies HPA behavior policies for scale up/down.
- **stabilizeRecommendation**: Prevents thrashing by considering recent recommendations.

## Supported Metric Types

- Resource metrics (CPU, memory utilization)
- Container resource metrics
- Pod metrics (custom metrics per pod)
- Object metrics (metrics from other Kubernetes objects)
- External metrics (from external systems)

## Design Patterns

- Uses downscale stabilization window to prevent rapid scale-down.
- Supports configurable scale-up/down behaviors with rate limits.
- Validates that pods aren't controlled by multiple HPAs.
- Exposes detailed Prometheus metrics via the monitor package.
- Implements tolerance-based scaling to avoid oscillation.
