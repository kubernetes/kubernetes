# Package: metrics

## Purpose
Defines and registers Prometheus metrics for the Ephemeral Volume controller to track PVC creation operations.

## Key Constants

- **EphemeralVolumeSubsystem**: Subsystem name "ephemeral_volume_controller" used for metric naming.

## Key Metrics

- **EphemeralVolumeCreateAttempts**: Counter tracking total PVC creation requests (both successful and unsuccessful).
  - Name: `ephemeral_volume_controller_create_total`
  - Stability level: ALPHA
- **EphemeralVolumeCreateFailures**: Counter tracking failed PVC creation requests.
  - Name: `ephemeral_volume_controller_create_failures_total`
  - Stability level: ALPHA

## Key Functions

- **RegisterMetrics()**: Thread-safe registration of metrics with the legacy registry (called once via sync.Once).

## Design Notes

- Uses `k8s.io/component-base/metrics` for Kubernetes-compatible metric definitions.
- Metrics are registered with the legacy registry for backward compatibility.
- Success rate can be calculated as: (attempts - failures) / attempts.
