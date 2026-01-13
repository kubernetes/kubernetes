# Package: monitor

Provides Prometheus metrics monitoring for the HPA controller.

## Key Types

- **Monitor**: Interface for observing HPA controller operations.
- **ActionLabel**: Labels for scale actions (scale_up, scale_down, none).
- **ErrorLabel**: Labels for error types (spec, internal, none).

## Key Interface Methods

- **ObserveReconciliationResult**: Records reconciliation outcome with action and error labels.
- **ObserveMetricComputationResult**: Records metric computation results by type.
- **ObserveHPAAddition/ObserveHPADeletion**: Tracks HPA object count.
- **ObserveDesiredReplicas**: Records the desired replica count per HPA.

## Key Metrics (from metrics.go)

- `reconciliationsTotal`: Counter of HPA reconciliations by action/error.
- `reconciliationsDuration`: Histogram of reconciliation duration.
- `metricComputationTotal`: Counter of metric computations by type.
- `metricComputationDuration`: Histogram of metric computation duration.
- `numHorizontalPodAutoscalers`: Gauge of total HPAs.
- `desiredReplicasCount`: Gauge of desired replicas per HPA.

## Design Patterns

- Separates monitoring interface from implementation for testability.
- Uses labeled metrics for detailed observability.
- Distinguishes spec errors (user mistakes) from internal errors.
