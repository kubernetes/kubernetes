# Package: kernel

## Purpose
Provides utilities for detecting Linux kernel version and constants for kernel version requirements of various features.

## Key Functions
- `GetVersion()` - Returns the currently running kernel version by reading /proc/sys/kernel/osrelease

## Key Version Constants
- `IPLocalReservedPortsNamespacedKernelVersion` (3.16) - When ip_local_reserved_ports became namespaced
- `IPVSConnReuseModeMinSupportedKernelVersion` (4.1) - Minimum for IPVS conn_reuse_mode
- `TCPKeepAliveTimeNamespacedKernelVersion` (4.5) - When TCP keepalive settings became namespaced
- `IPVSConnReuseModeFixedKernelVersion` (5.9) - When IPVS conn_reuse_mode was fixed
- `NFTablesKubeProxyKernelVersion` (5.13) - Minimum for nftables kube-proxy mode
- `TmpfsNoswapSupportKernelVersion` (6.4) - When tmpfs noswap was added

## Design Patterns
- Version parsing using k8s.io/apimachinery/pkg/util/version
- Constants document Linux kernel commit references
- Used for feature detection and compatibility checks
