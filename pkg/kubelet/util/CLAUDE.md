# Package: util

## Purpose
The `util` package provides miscellaneous utility functions for the Kubelet, including hostname handling, resource limits calculation, boot time detection, and startup latency tracking.

## Key Functions

- **FromApiserverCache**: Modifies GetOptions to serve GET requests from apiserver cache instead of etcd.
- **GetNodenameForKernel**: Computes the hostname for the kernel's nodename field, optionally using FQDN (max 64 chars).
- **GetContainerByIndex**: Safely extracts a container from spec/status arrays by index with validation.
- **GetLimits**: Calculates effective resource limits by merging container and pod-level limits.
- **IsUnixDomainSocket**: Checks if a path is a Unix domain socket (delegated to filesystem package).

## Platform-Specific Functions

- **GetBootTime**: Returns system boot time (Linux reads /proc/stat, Darwin uses sysctl, FreeBSD uses sysctl).

## Key Types

- **ResourceOpts**: Holds pod and container resource requirements for limit calculation.

## Subpackages

This package has several subpackages:
- `cache`: Object caching utilities
- `env`: Environment variable utilities
- `format`: Pod formatting utilities
- `ioutils`: I/O utility functions
- `manager`: Secret/ConfigMap managers (cache-based and watch-based)
- `queue`: Work queue implementations
- `sliceutils`: Slice manipulation utilities
- `store`: File-based key-value store
- `swap`: Swap configuration utilities

## Design Notes

- FQDN hostname is limited to 64 characters per Linux kernel nodename field specification.
- Resource limits calculation supports pod-level resource management (PodLevelResources feature).
- Boot time detection is platform-specific with separate implementations for Linux, Darwin, and FreeBSD.
