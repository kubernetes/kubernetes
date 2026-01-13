# Package: metrics

Prometheus metrics for the Job controller.

## Key Metrics

- `job_sync_duration_seconds`: Histogram of job sync latency (labels: completion_mode, result, action)
- `job_syncs_total`: Counter of job syncs (labels: completion_mode, result, action)
- `jobs_finished_total`: Counter of finished jobs (labels: completion_mode, result, reason)
- `job_pods_finished_total`: Counter of finished pods tracked (labels: completion_mode, result)
- `pod_failures_handled_by_failure_policy_total`: Counter by failure policy action
- `terminated_pods_tracking_finalizer_total`: Counter of finalizer additions/removals
- `job_finished_indexes_total`: Counter of finished indexes (labels: status, backoffLimit)
- `job_pods_creation_total`: Counter of pod creations (labels: reason, status)
- `jobs_by_external_controller_total`: Counter of externally managed jobs

## Key Constants

- Action labels: `reconciling`, `tracking`, `pods_created`, `pods_deleted`
- Result labels: `succeeded`, `failed`

## Key Functions

- `Register()`: Registers all metrics with the legacy registry

## Purpose

Provides comprehensive observability into Job controller operation including sync performance, job completion, pod lifecycle, and failure handling.

## Design Notes

- Uses `job_controller` subsystem
- Several metrics are STABLE stability level
