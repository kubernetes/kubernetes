# Package: metrics

This package provides Prometheus metrics for monitoring kube-proxy performance and health.

## Key Metrics

- `kubeproxy_sync_proxy_rules_duration_seconds` - Histogram of rule sync latency
- `kubeproxy_sync_proxy_rules_last_timestamp_seconds` - Gauge of last successful sync time
- `kubeproxy_network_programming_duration_seconds` - Histogram of network programming latency
- `kubeproxy_sync_proxy_rules_iptables_total` - Counter of iptables operations
- `kubeproxy_sync_proxy_rules_endpoint_changes_total` - Counter of endpoint changes
- `kubeproxy_sync_proxy_rules_service_changes_total` - Counter of service changes

## Key Functions

- `Register()` - Registers all kube-proxy metrics with the Prometheus registry
- `SinceInSeconds()` - Helper for calculating durations for histograms

## Design Notes

- Follows Kubernetes metrics conventions (kubeproxy_ prefix)
- Provides observability into sync frequency and latency
- Helps identify performance issues with large numbers of services
- Supports both iptables and IPVS mode metrics
- Includes metrics for conntrack operations and health checks
