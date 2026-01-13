# Package: v1alpha1

This package provides the v1alpha1 versioned API types for kube-proxy configuration, including registration, defaults, and conversion logic.

## Key Types

This package primarily re-exports types from `k8s.io/kube-proxy/config/v1alpha1`:
- `KubeProxyConfiguration` - Main configuration struct for kube-proxy

## Key Functions

- `AddToScheme()` - Registers v1alpha1 types with a runtime.Scheme and sets up defaults
- `SetDefaults_KubeProxyConfiguration()` - Applies default values to KubeProxyConfiguration
- `SetDefaults_KubeProxyIPTablesConfiguration()` - Sets defaults for iptables mode
- `SetDefaults_KubeProxyIPVSConfiguration()` - Sets defaults for IPVS mode
- `SetDefaults_KubeProxyConntrackConfiguration()` - Sets defaults for conntrack settings

## Design Notes

- Follows Kubernetes API versioning conventions with external/internal type separation
- Default sync period is 15 seconds for iptables mode
- Default masquerade bit is 14, minimum sync period is 1 second
- Supports conntrack configuration with TCP timeouts (0 = use system defaults)
