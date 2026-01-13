# Metrics Proposed for Graduation (Issue #136107)

This table contains all metrics proposed for graduation from Alpha to Beta in issue #136107.

| Metric Name | File Path | Current Stability | Owning Component |
|-------------|-----------|-------------------|------------------|
| `apiserver_admission_webhook_rejection_count` | `staging/src/k8s.io/apiserver/pkg/admission/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_admission_webhook_request_total` | `staging/src/k8s.io/apiserver/pkg/admission/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_audit_error_total` | `staging/src/k8s.io/apiserver/pkg/audit/metrics.go` | ALPHA | apiserver |
| `apiserver_audit_event_total` | `staging/src/k8s.io/apiserver/pkg/audit/metrics.go` | ALPHA | apiserver |
| `apiserver_audit_level_total` | `staging/src/k8s.io/apiserver/pkg/audit/metrics.go` | ALPHA | apiserver |
| `apiserver_authorization_webhook_duration_seconds` | `staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_authorization_webhook_evaluations_fail_open_total` | `staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_authorization_webhook_evaluations_total` | `staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_client_certificate_expiration_seconds` | `staging/src/k8s.io/apiserver/pkg/authentication/request/x509/x509.go` | ALPHA | apiserver |
| `apiserver_egress_dialer_dial_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/server/egressselector/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_egress_dialer_dial_failure_count` | `staging/src/k8s.io/apiserver/pkg/server/egressselector/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_egress_dialer_dial_start_total` | `staging/src/k8s.io/apiserver/pkg/server/egressselector/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_envelope_encryption_dek_cache_fill_percent` | `staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_envelope_encryption_dek_cache_inter_arrival_time_seconds` | `staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics/metrics.go` | ALPHA | apiserver |
| `etcd_bookmark_counts` | `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_flowcontrol_priority_level_request_utilization` | `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_flowcontrol_priority_level_seat_utilization` | `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_flowcontrol_request_concurrency_in_use` | `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_flowcontrol_request_concurrency_limit` | `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_flowcontrol_request_execution_seconds` | `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_flowcontrol_work_estimated_seats` | `staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_kube_aggregator_x509_insecure_sha1_total` | `staging/src/k8s.io/kube-aggregator/pkg/apiserver/metrics.go` | ALPHA | kube-aggregator |
| `apiserver_kube_aggregator_x509_missing_san_total` | `staging/src/k8s.io/kube-aggregator/pkg/apiserver/metrics.go` | ALPHA | kube-aggregator |
| `kubernetes_build_info` | `staging/src/k8s.io/component-base/metrics/version.go` | ALPHA | component-base |
| `apiserver_request_filter_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_request_sli_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `rest_client_request_duration_seconds` | `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go` | ALPHA | component-base (restclient) |
| `rest_client_requests_total` | `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go` | ALPHA | component-base (restclient) |
| `serviceaccount_legacy_tokens_total` | `pkg/serviceaccount/metrics.go` | ALPHA | apiserver |
| `serviceaccount_stale_tokens_total` | `pkg/serviceaccount/metrics.go` | ALPHA | apiserver |
| `serviceaccount_valid_tokens_total` | `pkg/serviceaccount/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_data_key_generation_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/storage/value/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_data_key_generation_failures_total` | `staging/src/k8s.io/apiserver/pkg/storage/value/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_envelope_transformation_cache_misses_total` | `staging/src/k8s.io/apiserver/pkg/storage/value/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_events_received_total` | `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_transformation_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/storage/value/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_transformation_operations_total` | `staging/src/k8s.io/apiserver/pkg/storage/value/metrics.go` | ALPHA | apiserver |
| `apiserver_terminated_watchers_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `watch_cache_capacity` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_consistent_read_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_events_dispatched_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_initializations_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_read_wait_seconds` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_resource_version` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_events_sizes` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_events_total` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_webhooks_x509_insecure_sha1_total` | `staging/src/k8s.io/apiserver/pkg/util/webhook/metrics.go` | ALPHA | apiserver |
| `apiserver_webhooks_x509_missing_san_total` | `staging/src/k8s.io/apiserver/pkg/util/webhook/metrics.go` | ALPHA | apiserver |
| `endpoint_slice_controller_changes` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_desired_endpoint_slices` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_endpoints_added_per_sync` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_endpoints_desired` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_endpoints_removed_per_sync` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_num_endpoint_slices` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_services_count_by_traffic_distribution` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `job_controller_pod_failures_handled_by_failure_policy_total` | `pkg/controller/job/metrics/metrics.go` | ALPHA | kube-controller-manager (job) |
| `job_controller_terminated_pods_tracking_finalizer_total` | `pkg/controller/job/metrics/metrics.go` | ALPHA | kube-controller-manager (job) |
| `resourceclaim_controller_resource_claims` | `pkg/controller/resourceclaim/metrics/metrics.go` | ALPHA | kube-controller-manager (resourceclaim) |
| `running_managed_controllers` | `staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go` | ALPHA | component-base (controllers) |
| `storage_operation_duration_seconds` | `pkg/volume/util/metrics.go` | ALPHA | kubelet |
| `volume_operation_total_errors` | `pkg/controller/volume/persistentvolume/metrics/metrics.go` | ALPHA | kube-controller-manager (volume) |
| `volume_operation_total_seconds` | `pkg/volume/util/metrics.go` | ALPHA | kubelet |
| `scheduler_goroutines` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_permit_wait_duration_seconds` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_plugin_evaluation_total` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_plugin_execution_duration_seconds` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_scheduling_algorithm_duration_seconds` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_unschedulable_pods` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |

## Notes

- All metrics listed have **ALPHA** stability level (as required for graduation to Beta)
- File paths are relative to the Kubernetes repository root
- Component names indicate the owning Kubernetes component
- Some metrics may require additional verification of exact file locations
