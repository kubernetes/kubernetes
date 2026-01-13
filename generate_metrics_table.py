#!/usr/bin/env python3
"""
Generate comprehensive metrics table for issue #136107
"""
import subprocess
import re

# All metrics from issue
METRICS = [
    "apiserver_admission_webhook_rejection_count",
    "apiserver_admission_webhook_request_total",
    "apiserver_audit_error_total",
    "apiserver_audit_event_total",
    "apiserver_audit_level_total",
    "apiserver_authorization_webhook_duration_seconds",
    "apiserver_authorization_webhook_evaluations_fail_open_total",
    "apiserver_authorization_webhook_evaluations_total",
    "apiserver_client_certificate_expiration_seconds",
    "apiserver_egress_dialer_dial_duration_seconds",
    "apiserver_egress_dialer_dial_failure_count",
    "apiserver_egress_dialer_dial_start_total",
    "apiserver_envelope_encryption_dek_cache_fill_percent",
    "apiserver_envelope_encryption_dek_cache_inter_arrival_time_seconds",
    "etcd_bookmark_counts",
    "apiserver_flowcontrol_priority_level_request_utilization",
    "apiserver_flowcontrol_priority_level_seat_utilization",
    "apiserver_flowcontrol_request_concurrency_in_use",
    "apiserver_flowcontrol_request_concurrency_limit",
    "apiserver_flowcontrol_request_execution_seconds",
    "apiserver_flowcontrol_work_estimated_seats",
    "apiserver_kube_aggregator_x509_insecure_sha1_total",
    "apiserver_kube_aggregator_x509_missing_san_total",
    "kubernetes_build_info",
    "apiserver_request_filter_duration_seconds",
    "apiserver_request_sli_duration_seconds",
    "rest_client_request_duration_seconds",
    "rest_client_requests_total",
    "serviceaccount_legacy_tokens_total",
    "serviceaccount_stale_tokens_total",
    "serviceaccount_valid_tokens_total",
    "apiserver_storage_data_key_generation_duration_seconds",
    "apiserver_storage_data_key_generation_failures_total",
    "apiserver_storage_envelope_transformation_cache_misses_total",
    "apiserver_storage_events_received_total",
    "apiserver_storage_transformation_duration_seconds",
    "apiserver_storage_transformation_operations_total",
    "apiserver_terminated_watchers_total",
    "watch_cache_capacity",
    "apiserver_watch_cache_consistent_read_total",
    "apiserver_watch_cache_events_dispatched_total",
    "apiserver_watch_cache_initializations_total",
    "apiserver_watch_cache_read_wait_seconds",
    "apiserver_watch_cache_resource_version",
    "apiserver_watch_events_sizes",
    "apiserver_watch_events_total",
    "apiserver_webhooks_x509_insecure_sha1_total",
    "apiserver_webhooks_x509_missing_san_total",
    "endpoint_slice_controller_changes",
    "endpoint_slice_controller_desired_endpoint_slices",
    "endpoint_slice_controller_endpoints_added_per_sync",
    "endpoint_slice_controller_endpoints_desired",
    "endpoint_slice_controller_endpoints_removed_per_sync",
    "endpoint_slice_controller_num_endpoint_slices",
    "endpoint_slice_controller_services_count_by_traffic_distribution",
    "job_controller_pod_failures_handled_by_failure_policy_total",
    "job_controller_terminated_pods_tracking_finalizer_total",
    "resourceclaim_controller_resource_claims",
    "running_managed_controllers",
    "storage_operation_duration_seconds",
    "volume_operation_total_errors",
    "volume_operation_total_seconds",
    "scheduler_goroutines",
    "scheduler_permit_wait_duration_seconds",
    "scheduler_plugin_evaluation_total",
    "scheduler_plugin_execution_duration_seconds",
    "scheduler_scheduling_algorithm_duration_seconds",
    "scheduler_unschedulable_pods",
]

# Known file locations based on our searches
KNOWN_FILES = {
    "staging/src/k8s.io/apiserver/pkg/admission/metrics/metrics.go",
    "staging/src/k8s.io/apiserver/pkg/audit/metrics.go",
    "staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook/metrics/metrics.go",
    "pkg/serviceaccount/metrics.go",
    "staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go",
    "staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go",
    "staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go",
    "staging/src/k8s.io/apiserver/pkg/storage/cacher/metrics/metrics.go",
    "staging/src/k8s.io/endpointslice/metrics/metrics.go",
    "pkg/controller/job/metrics/metrics.go",
    "staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go",
    "pkg/volume/util/metrics.go",
    "pkg/controller/volume/persistentvolume/metrics/metrics.go",
    "pkg/scheduler/metrics/metrics.go",
}

def search_metric_in_files(metric_name):
    """Search for metric in known files"""
    # Extract name part (after last _ that matches known patterns)
    name_parts = metric_name.split('_')
    
    # Try different name patterns
    patterns_to_try = []
    
    # Full name without prefix
    if metric_name.startswith("apiserver_"):
        patterns_to_try.append(metric_name[10:])  # Remove "apiserver_"
    if metric_name.startswith("scheduler_"):
        patterns_to_try.append(metric_name[10:])  # Remove "scheduler_"
    if metric_name.startswith("endpoint_slice_controller_"):
        patterns_to_try.append(metric_name[26:])  # Remove prefix
    if metric_name.startswith("job_controller_"):
        patterns_to_try.append(metric_name[15:])  # Remove prefix
    
    # Also try last 2-3 parts
    if len(name_parts) >= 2:
        patterns_to_try.append("_".join(name_parts[-2:]))
    if len(name_parts) >= 3:
        patterns_to_try.append("_".join(name_parts[-3:]))
    
    patterns_to_try.append(name_parts[-1])  # Last part
    
    for pattern in patterns_to_try:
        search_term = f'Name:\\s*"{re.escape(pattern)}"'
        for filepath in KNOWN_FILES:
            try:
                result = subprocess.run(
                    ['grep', '-l', search_term, filepath],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and filepath in result.stdout:
                    return filepath, pattern
            except:
                continue
    
    return None, None

def get_stability(filepath, name_part):
    """Extract stability level from file"""
    if not filepath:
        return "ALPHA"
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        pattern = rf'Name:\s*"{re.escape(name_part)}".*?StabilityLevel:\s*metrics\.(\w+)'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        return match.group(1) if match else "ALPHA"
    except:
        return "ALPHA"

def get_component(filepath):
    """Determine component from filepath"""
    if not filepath:
        return "Unknown"
    if 'apiserver' in filepath:
        return "apiserver"
    elif 'scheduler' in filepath:
        return "scheduler"
    elif 'controller' in filepath:
        if 'job' in filepath:
            return "kube-controller-manager (job)"
        elif 'endpoint' in filepath:
            return "kube-controller-manager (endpointslice)"
        elif 'volume' in filepath:
            return "kube-controller-manager (volume)"
        elif 'resourceclaim' in filepath:
            return "kube-controller-manager (resourceclaim)"
        return "kube-controller-manager"
    elif 'volume' in filepath and 'util' in filepath:
        return "kubelet"
    elif 'serviceaccount' in filepath:
        return "apiserver"
    elif 'component-base' in filepath:
        if 'restclient' in filepath:
            return "component-base (restclient)"
        elif 'controllers' in filepath:
            return "component-base (controllers)"
        return "component-base"
    elif 'endpointslice' in filepath:
        return "kube-controller-manager (endpointslice)"
    return "Unknown"

print("| Metric Name | File Path | Current Stability | Owning Component |")
print("|-------------|-----------|-------------------|------------------|")

for metric in METRICS:
    filepath, name_part = search_metric_in_files(metric)
    stability = get_stability(filepath, name_part) if filepath else "ALPHA"
    component = get_component(filepath) if filepath else "Unknown"
    
    file_display = filepath if filepath else "NOT FOUND"
    if len(file_display) > 70:
        file_display = file_display[:67] + "..."
    
    print(f"| `{metric}` | `{file_display}` | {stability} | {component} |")
