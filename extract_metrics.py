#!/usr/bin/env python3
"""
Script to extract metric information from Kubernetes codebase for issue #136107
"""
import subprocess
import re
import json
from pathlib import Path

# Metrics from issue #136107
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

def extract_name_part(metric_name):
    """Extract the name part from full metric name"""
    # Remove common prefixes
    prefixes = [
        "apiserver_",
        "scheduler_",
        "endpoint_slice_controller_",
        "job_controller_",
        "kubernetes_",
    ]
    name = metric_name
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name

def find_metric_file(metric_name, name_part):
    """Find the file containing the metric definition"""
    # Try searching for Name: "name_part"
    pattern = f'Name:\\s*"{re.escape(name_part)}"'
    try:
        result = subprocess.run(
            ['grep', '-r', '--include=*.go', '-l', pattern, '.'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd='.'
        )
        if result.returncode == 0 and result.stdout:
            files = result.stdout.strip().split('\n')
            # Filter out test files, prefer metrics.go files
            for f in files:
                if 'metrics.go' in f and '_test.go' not in f:
                    return f.strip('./')
            # If no metrics.go, return first non-test file
            for f in files:
                if '_test.go' not in f:
                    return f.strip('./')
            return files[0].strip('./') if files else None
    except:
        pass
    return None

def extract_stability_and_component(filepath, name_part):
    """Extract stability level and determine component from file"""
    if not filepath:
        return None, None, None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Find the metric definition block
        # Look for Name: "name_part" and nearby StabilityLevel
        pattern = rf'Name:\s*"{re.escape(name_part)}"[^{{]*?StabilityLevel:\s*metrics\.(\w+)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            stability = match.group(1)
        else:
            # Try alternative pattern
            pattern2 = rf'Name:\s*"{re.escape(name_part)}".*?StabilityLevel:\s*metrics\.(\w+)'
            match = re.search(pattern2, content, re.MULTILINE | re.DOTALL)
            stability = match.group(1) if match else "ALPHA"  # Default assumption
        
        # Determine component from file path
        component = "Unknown"
        if 'apiserver' in filepath:
            component = "apiserver"
        elif 'scheduler' in filepath:
            component = "scheduler"
        elif 'controller' in filepath or 'endpointslice' in filepath:
            if 'job' in filepath:
                component = "kube-controller-manager (job)"
            elif 'endpoint' in filepath:
                component = "kube-controller-manager (endpointslice)"
            elif 'resourceclaim' in filepath:
                component = "kube-controller-manager (resourceclaim)"
            else:
                component = "kube-controller-manager"
        elif 'volume' in filepath:
            component = "kubelet"
        elif 'component-base' in filepath:
            if 'restclient' in filepath:
                component = "component-base (restclient)"
            elif 'controllers' in filepath:
                component = "component-base (controllers)"
            else:
                component = "component-base"
        
        return filepath, stability, component
    except Exception as e:
        return filepath, "ALPHA", "Unknown"

def main():
    results = []
    for metric in METRICS:
        name_part = extract_name_part(metric)
        filepath = find_metric_file(metric, name_part)
        filepath, stability, component = extract_stability_and_component(filepath, name_part)
        
        results.append({
            'metric': metric,
            'file_path': filepath or "NOT FOUND",
            'stability': stability or "ALPHA",
            'component': component or "Unknown"
        })
        print(f"Processed: {metric}")
    
    # Output as JSON
    print("\n" + "="*80)
    print(json.dumps(results, indent=2))
    
    # Also create markdown table
    print("\n" + "="*80)
    print("\n## Metrics Table\n")
    print("| Metric Name | File Path | Current Stability | Owning Component |")
    print("|-------------|-----------|-------------------|------------------|")
    for r in results:
        file_display = r['file_path'][:60] + "..." if len(r['file_path']) > 60 else r['file_path']
        print(f"| `{r['metric']}` | `{file_display}` | {r['stability']} | {r['component']} |")

if __name__ == "__main__":
    main()
