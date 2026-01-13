# Package: metrics

## Purpose
The `metrics` package provides Prometheus metrics for monitoring volume manager operations and volume reconstruction.

## Key Metrics

- **volume_manager_selinux_container_errors_total**: Counter for errors when kubelet cannot compute SELinux context for a container.
- **volume_manager_selinux_container_warnings_total**: Counter for warnings about SELinux context issues.
- **volume_manager_selinux_pod_context_mismatch_errors_total**: Counter for errors when volumes have conflicting SELinux contexts.
- **volume_manager_selinux_pod_context_mismatch_warnings_total**: Counter for warnings about SELinux context mismatches.
- **volume_manager_selinux_volume_context_mismatch_errors_total**: Counter for errors when existing volume mount has different SELinux context.
- **volume_manager_selinux_volume_context_mismatch_warnings_total**: Counter for warnings about volume context mismatches.
- **reconstruct_volume_operations_total**: Counter for volume reconstruction operations during kubelet startup.
- **reconstruct_volume_operations_errors_total**: Counter for failed volume reconstruction operations.
- **force_cleaned_failed_volume_operations_total**: Counter for forced cleanup of failed volume operations.
- **force_cleaned_failed_volume_operation_errors_total**: Counter for errors during forced cleanup.

## Design Notes

- Metrics are registered with the legacyregistry for global access.
- SELinux metrics help diagnose mount context issues.
- Reconstruction metrics track kubelet restart recovery.
- All metrics follow Kubernetes stability conventions.
