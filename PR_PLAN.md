# PR Plan for Metrics Graduation (Alpha → Beta)

This document groups the 68 metrics from issue #136107 into 5 PRs for graduation from Alpha to Beta.

## Alpha → Beta Graduation Requirements

Based on the graduation criteria from `METRICS_GRADUATION_SUMMARY.md`, each PR must ensure:

1. ✅ **Testing Requirement**: The metric must have a corresponding test that validates:
   - The metric is registered and emitted correctly
   - The metric has the expected labels and values under known conditions

2. ✅ **Documentation**: Ensure the metric has a clear and accurate help text description

3. ✅ **Component Owner Acknowledgment**: Get component owner acknowledgment for supporting the metric across multiple releases (in accordance with the metric stability policy)

4. ⚠️ **API Review**: API review process specifically targets stable metrics. Component owner acknowledgment is definitely required.

5. ✅ **Understand API Guarantees**: Once promoted to Beta, labels cannot be removed (but can be added), providing stability for existing dashboards/alerts

---

## PR 1: component-base Metrics

**Component**: component-base

### Metrics (4 metrics)

1. `kubernetes_build_info`
2. `rest_client_request_duration_seconds`
3. `rest_client_requests_total`
4. `running_managed_controllers`

### File Paths

- `staging/src/k8s.io/component-base/metrics/version.go`
- `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go`
- `staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go`

### Checklist

- [ ] **Testing Requirement**: Verify each metric has tests validating:
  - [ ] Metric is registered and emitted correctly
  - [ ] Metric has expected labels and values under known conditions
- [ ] **Documentation**: Verify each metric has clear and accurate help text
- [ ] **Component Owner Acknowledgment**: Get component-base owner acknowledgment for supporting these metrics across multiple releases
- [ ] **Code Changes**: Update `StabilityLevel: metrics.ALPHA` to `StabilityLevel: metrics.BETA` in all relevant files
- [ ] **Review**: Ensure all changes align with metric stability policy

---

## PR 2: scheduler Metrics

**Component**: scheduler

### Metrics (6 metrics)

1. `scheduler_goroutines`
2. `scheduler_permit_wait_duration_seconds`
3. `scheduler_plugin_evaluation_total`
4. `scheduler_plugin_execution_duration_seconds`
5. `scheduler_scheduling_algorithm_duration_seconds`
6. `scheduler_unschedulable_pods`

### File Paths

- `pkg/scheduler/metrics/metrics.go`

### Checklist

- [ ] **Testing Requirement**: Verify each metric has tests validating:
  - [ ] Metric is registered and emitted correctly
  - [ ] Metric has expected labels and values under known conditions
- [ ] **Documentation**: Verify each metric has clear and accurate help text
- [ ] **Component Owner Acknowledgment**: Get scheduler component owner acknowledgment for supporting these metrics across multiple releases
- [ ] **Code Changes**: Update `StabilityLevel: metrics.ALPHA` to `StabilityLevel: metrics.BETA` for all 6 metrics in `pkg/scheduler/metrics/metrics.go`
- [ ] **Review**: Ensure all changes align with metric stability policy

---

## PR 3: kubelet Volume Metrics

**Component**: kubelet

### Metrics (2 metrics)

1. `storage_operation_duration_seconds`
2. `volume_operation_total_seconds`

### File Paths

- `pkg/volume/util/metrics.go`

### Checklist

- [ ] **Testing Requirement**: Verify each metric has tests validating:
  - [ ] Metric is registered and emitted correctly
  - [ ] Metric has expected labels and values under known conditions
- [ ] **Documentation**: Verify each metric has clear and accurate help text
- [ ] **Component Owner Acknowledgment**: Get kubelet component owner acknowledgment for supporting these metrics across multiple releases
- [ ] **Code Changes**: Update `StabilityLevel: metrics.ALPHA` to `StabilityLevel: metrics.BETA` for both metrics in `pkg/volume/util/metrics.go`
- [ ] **Review**: Ensure all changes align with metric stability policy

---

## PR 4: kube-controller-manager Metrics

**Component**: kube-controller-manager

### Metrics (11 metrics)

1. `endpoint_slice_controller_changes`
2. `endpoint_slice_controller_desired_endpoint_slices`
3. `endpoint_slice_controller_endpoints_added_per_sync`
4. `endpoint_slice_controller_endpoints_desired`
5. `endpoint_slice_controller_endpoints_removed_per_sync`
6. `endpoint_slice_controller_num_endpoint_slices`
7. `endpoint_slice_controller_services_count_by_traffic_distribution`
8. `job_controller_pod_failures_handled_by_failure_policy_total`
9. `job_controller_terminated_pods_tracking_finalizer_total`
10. `resourceclaim_controller_resource_claims`
11. `volume_operation_total_errors`

### File Paths

- `staging/src/k8s.io/endpointslice/metrics/metrics.go`
- `pkg/controller/job/metrics/metrics.go`
- `pkg/controller/resourceclaim/metrics/metrics.go`
- `pkg/controller/volume/persistentvolume/metrics/metrics.go`

### Checklist

- [ ] **Testing Requirement**: Verify each metric has tests validating:
  - [ ] Metric is registered and emitted correctly
  - [ ] Metric has expected labels and values under known conditions
- [ ] **Documentation**: Verify each metric has clear and accurate help text
- [ ] **Component Owner Acknowledgment**: Get kube-controller-manager component owner acknowledgment for supporting these metrics across multiple releases
- [ ] **Code Changes**: Update `StabilityLevel: metrics.ALPHA` to `StabilityLevel: metrics.BETA` in all relevant files:
  - [ ] `staging/src/k8s.io/endpointslice/metrics/metrics.go` (7 metrics)
  - [ ] `pkg/controller/job/metrics/metrics.go` (2 metrics)
  - [ ] `pkg/controller/resourceclaim/metrics/metrics.go` (1 metric)
  - [ ] `pkg/controller/volume/persistentvolume/metrics/metrics.go` (1 metric)
- [ ] **Review**: Ensure all changes align with metric stability policy

---

## PR 5: apiserver + kube-aggregator Metrics

**Components**: apiserver, kube-aggregator

### Metrics (45 metrics)

**Admission Metrics (2):**
1. `apiserver_admission_webhook_rejection_count`
2. `apiserver_admission_webhook_request_total`

**Audit Metrics (3):**
3. `apiserver_audit_error_total`
4. `apiserver_audit_event_total`
5. `apiserver_audit_level_total`

**Authorization Metrics (3):**
6. `apiserver_authorization_webhook_duration_seconds`
7. `apiserver_authorization_webhook_evaluations_fail_open_total`
8. `apiserver_authorization_webhook_evaluations_total`

**Authentication Metrics (1):**
9. `apiserver_client_certificate_expiration_seconds`

**Egress Dialer Metrics (3):**
10. `apiserver_egress_dialer_dial_duration_seconds`
11. `apiserver_egress_dialer_dial_failure_count`
12. `apiserver_egress_dialer_dial_start_total`

**Envelope Encryption Metrics (2):**
13. `apiserver_envelope_encryption_dek_cache_fill_percent`
14. `apiserver_envelope_encryption_dek_cache_inter_arrival_time_seconds`

**ETCD Metrics (1):**
15. `etcd_bookmark_counts`

**Flow Control Metrics (6):**
16. `apiserver_flowcontrol_priority_level_request_utilization`
17. `apiserver_flowcontrol_priority_level_seat_utilization`
18. `apiserver_flowcontrol_request_concurrency_in_use`
19. `apiserver_flowcontrol_request_concurrency_limit`
20. `apiserver_flowcontrol_request_execution_seconds`
21. `apiserver_flowcontrol_work_estimated_seats`

**Kube-Aggregator Metrics (2):**
22. `apiserver_kube_aggregator_x509_insecure_sha1_total`
23. `apiserver_kube_aggregator_x509_missing_san_total`

**Request Metrics (2):**
24. `apiserver_request_filter_duration_seconds`
25. `apiserver_request_sli_duration_seconds`

**Service Account Metrics (3):**
26. `serviceaccount_legacy_tokens_total`
27. `serviceaccount_stale_tokens_total`
28. `serviceaccount_valid_tokens_total`

**Storage Metrics (6):**
29. `apiserver_storage_data_key_generation_duration_seconds`
30. `apiserver_storage_data_key_generation_failures_total`
31. `apiserver_storage_envelope_transformation_cache_misses_total`
32. `apiserver_storage_events_received_total`
33. `apiserver_storage_transformation_duration_seconds`
34. `apiserver_storage_transformation_operations_total`

**Watch Cache Metrics (9):**
35. `apiserver_terminated_watchers_total`
36. `watch_cache_capacity`
37. `apiserver_watch_cache_consistent_read_total`
38. `apiserver_watch_cache_events_dispatched_total`
39. `apiserver_watch_cache_initializations_total`
40. `apiserver_watch_cache_read_wait_seconds`
41. `apiserver_watch_cache_resource_version`
42. `apiserver_watch_events_sizes`
43. `apiserver_watch_events_total`

**Webhook X509 Metrics (2):**
44. `apiserver_webhooks_x509_insecure_sha1_total`
45. `apiserver_webhooks_x509_missing_san_total`

### File Paths

- `staging/src/k8s.io/apiserver/pkg/admission/metrics/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/audit/metrics.go`
- `staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/authentication/request/x509/x509.go`
- `staging/src/k8s.io/apiserver/pkg/server/egressselector/metrics/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go`
- `staging/src/k8s.io/kube-aggregator/pkg/apiserver/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go`
- `pkg/serviceaccount/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/storage/value/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go`
- `staging/src/k8s.io/apiserver/pkg/util/webhook/metrics.go`

### Checklist

- [ ] **Testing Requirement**: Verify each metric has tests validating:
  - [ ] Metric is registered and emitted correctly
  - [ ] Metric has expected labels and values under known conditions
- [ ] **Documentation**: Verify each metric has clear and accurate help text
- [ ] **Component Owner Acknowledgment**: Get apiserver and kube-aggregator component owner acknowledgment for supporting these metrics across multiple releases
- [ ] **Code Changes**: Update `StabilityLevel: metrics.ALPHA` to `StabilityLevel: metrics.BETA` in all relevant files:
  - [ ] `staging/src/k8s.io/apiserver/pkg/admission/metrics/metrics.go` (2 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/audit/metrics.go` (3 metrics)
  - [ ] `staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/metrics.go` (3 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/authentication/request/x509/x509.go` (1 metric)
  - [ ] `staging/src/k8s.io/apiserver/pkg/server/egressselector/metrics/metrics.go` (3 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics/metrics.go` (2 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go` (1 metric)
  - [ ] `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go` (6 metrics)
  - [ ] `staging/src/k8s.io/kube-aggregator/pkg/apiserver/metrics.go` (2 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` (4 metrics)
  - [ ] `pkg/serviceaccount/metrics.go` (3 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/storage/value/metrics.go` (5 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` (7 metrics)
  - [ ] `staging/src/k8s.io/apiserver/pkg/util/webhook/metrics.go` (2 metrics)
- [ ] **Review**: Ensure all changes align with metric stability policy

---

## Summary

| PR | Component | Metrics Count | Files Count |
|----|-----------|---------------|-------------|
| PR 1 | component-base | 4 | 3 |
| PR 2 | scheduler | 6 | 1 |
| PR 3 | kubelet (volume) | 2 | 1 |
| PR 4 | kube-controller-manager | 11 | 4 |
| PR 5 | apiserver + kube-aggregator | 45 | 14 |
| **Total** | | **68** | **23** |
