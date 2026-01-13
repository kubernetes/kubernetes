# Package: ports

## Purpose
Defines default port constants for all major Kubernetes cluster components.

## Key Constants
- `ProxyStatusPort` (10249): Default port for kube-proxy metrics server
- `KubeletPort` (10250): Default port for kubelet server
- `KubeletReadOnlyPort` (10255): Read-only kubelet port for monitoring (legacy heapster support)
- `KubeletHealthzPort` (10248): Kubelet health check endpoint
- `ProxyHealthzPort` (10256): Kube-proxy health check endpoint
- `KubeControllerManagerPort` (10257): Controller manager status server
- `CloudControllerManagerPort` (10258): Cloud controller manager server
- `CloudControllerManagerWebhookPort`: Cloud controller manager webhook server

## Design Notes
- All ports may be overridden by command-line flags at startup
- Serves as central documentation of default ports used in the cluster
- Important for firewall rules, security policies, and monitoring configurations
- KubeletReadOnlyPort is considered legacy and should be avoided in favor of secure endpoints
