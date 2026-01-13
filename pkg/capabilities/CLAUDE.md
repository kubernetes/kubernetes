# Package: capabilities

## Purpose
Provides a global singleton for managing Kubernetes system capabilities that control privileged operations.

## Key Types/Structs
- `Capabilities`: Main struct containing AllowPrivileged flag, PrivilegedSources, and PerConnectionBandwidthLimitBytesPerSec
- `PrivilegedSources`: Defines pod sources allowed for host networking, host PID, and host IPC namespaces

## Key Functions
- `Initialize(c Capabilities)`: One-time initialization of capabilities (subsequent calls are ignored)
- `Setup(allowPrivileged, perConnectionBytesPerSec)`: Convenience wrapper for Initialize
- `Get() Capabilities`: Returns a read-only copy of current capabilities
- `ResetForTest()`: Resets capabilities state for testing purposes

## Key Capability Fields
- `AllowPrivileged`: Whether privileged containers are allowed
- `PrivilegedSources.HostNetworkSources`: Pod sources allowed to use host networking
- `PrivilegedSources.HostPIDSources`: Pod sources allowed to share host PID namespace
- `PrivilegedSources.HostIPCSources`: Pod sources allowed to share host IPC namespace
- `PerConnectionBandwidthLimitBytesPerSec`: Throughput limit for proxy/exec/attach connections

## Design Notes
- Uses sync.Once to ensure single initialization
- Global singleton pattern with thread-safe access via sync.Mutex
- Capabilities are currently global but may become per-user in the future
