# Package: validation

This package provides validation logic for KubeProxyConfiguration objects, ensuring configuration values are valid before kube-proxy starts.

## Key Functions

- `Validate()` - Main validation entry point for KubeProxyConfiguration
- `validateKubeProxyIPTablesConfiguration()` - Validates iptables-specific settings
- `validateKubeProxyIPVSConfiguration()` - Validates IPVS-specific settings
- `validateKubeProxyConntrackConfiguration()` - Validates conntrack timeout settings
- `validateProxyMode()` - Ensures proxy mode is valid for the platform
- `validateClientConnectionConfiguration()` - Validates kubeconfig and connection settings
- `validateHostPort()` - Validates host:port format for health/metrics endpoints

## Design Notes

- Uses Kubernetes field.ErrorList pattern for accumulating validation errors
- Validates durations are non-negative and within acceptable ranges
- Validates IP addresses and CIDRs for proper format
- Platform-aware validation (e.g., different valid modes for Linux vs Windows)
- Validates feature gate dependencies (e.g., LocalStorageCapacityIsolation)
