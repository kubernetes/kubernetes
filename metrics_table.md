| Metric Name | File Path | Current Stability | Owning Component |
|-------------|-----------|-------------------|------------------|
| `apiserver_admission_webhook_rejection_count` | `staging/src/k8s.io/apiserver/pkg/admission/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_admission_webhook_request_total` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_audit_error_total` | `staging/src/k8s.io/apiserver/pkg/audit/metrics.go` | ALPHA | apiserver |
| `apiserver_audit_event_total` | `staging/src/k8s.io/apiserver/pkg/audit/metrics.go` | ALPHA | apiserver |
| `apiserver_audit_level_total` | `staging/src/k8s.io/apiserver/pkg/audit/metrics.go` | ALPHA | apiserver |
| `apiserver_authorization_webhook_duration_seconds` | `staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/...` | ALPHA | apiserver |
| `apiserver_authorization_webhook_evaluations_fail_open_total` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_authorization_webhook_evaluations_total` | `staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/...` | ALPHA | apiserver |
| `apiserver_client_certificate_expiration_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_egress_dialer_dial_duration_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_egress_dialer_dial_failure_count` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_egress_dialer_dial_start_total` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_envelope_encryption_dek_cache_fill_percent` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_envelope_encryption_dek_cache_inter_arrival_time_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `etcd_bookmark_counts` | `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_flowcontrol_priority_level_request_utilization` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_flowcontrol_priority_level_seat_utilization` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_flowcontrol_request_concurrency_in_use` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_flowcontrol_request_concurrency_limit` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_flowcontrol_request_execution_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_flowcontrol_work_estimated_seats` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_kube_aggregator_x509_insecure_sha1_total` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_kube_aggregator_x509_missing_san_total` | `NOT FOUND` | ALPHA | Unknown |
| `kubernetes_build_info` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_request_filter_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_request_sli_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `rest_client_request_duration_seconds` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `rest_client_requests_total` | `NOT FOUND` | ALPHA | Unknown |
| `serviceaccount_legacy_tokens_total` | `pkg/serviceaccount/metrics.go` | ALPHA | apiserver |
| `serviceaccount_stale_tokens_total` | `pkg/serviceaccount/metrics.go` | ALPHA | apiserver |
| `serviceaccount_valid_tokens_total` | `pkg/serviceaccount/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_data_key_generation_duration_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_storage_data_key_generation_failures_total` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_storage_envelope_transformation_cache_misses_total` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_storage_events_received_total` | `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_storage_transformation_duration_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_storage_transformation_operations_total` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_terminated_watchers_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `watch_cache_capacity` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_consistent_read_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_events_dispatched_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_initializations_total` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_read_wait_seconds` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_cache_resource_version` | `staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_events_sizes` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_watch_events_total` | `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go` | ALPHA | apiserver |
| `apiserver_webhooks_x509_insecure_sha1_total` | `NOT FOUND` | ALPHA | Unknown |
| `apiserver_webhooks_x509_missing_san_total` | `NOT FOUND` | ALPHA | Unknown |
| `endpoint_slice_controller_changes` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_desired_endpoint_slices` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_endpoints_added_per_sync` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_endpoints_desired` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_endpoints_removed_per_sync` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_num_endpoint_slices` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `endpoint_slice_controller_services_count_by_traffic_distribution` | `staging/src/k8s.io/endpointslice/metrics/metrics.go` | ALPHA | kube-controller-manager (endpointslice) |
| `job_controller_pod_failures_handled_by_failure_policy_total` | `pkg/controller/job/metrics/metrics.go` | ALPHA | kube-controller-manager (job) |
| `job_controller_terminated_pods_tracking_finalizer_total` | `pkg/controller/job/metrics/metrics.go` | ALPHA | kube-controller-manager (job) |
| `resourceclaim_controller_resource_claims` | `NOT FOUND` | ALPHA | Unknown |
| `running_managed_controllers` | `staging/src/k8s.io/component-base/metrics/prometheus/controllers/me...` | ALPHA | kube-controller-manager |
| `storage_operation_duration_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `volume_operation_total_errors` | `NOT FOUND` | ALPHA | Unknown |
| `volume_operation_total_seconds` | `NOT FOUND` | ALPHA | Unknown |
| `scheduler_goroutines` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_permit_wait_duration_seconds` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_plugin_evaluation_total` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_plugin_execution_duration_seconds` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_scheduling_algorithm_duration_seconds` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
| `scheduler_unschedulable_pods` | `pkg/scheduler/metrics/metrics.go` | ALPHA | scheduler |
