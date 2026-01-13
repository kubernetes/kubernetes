# Package: componentstatus

## Purpose
Implements a read-only REST endpoint for ComponentStatus, which reports the health of control plane components (etcd, scheduler, controller-manager).

## Key Types

- **REST**: REST storage that dynamically probes component health (no backing store)
- **Server**: Interface for health check implementations
- **HttpServer**: HTTP/HTTPS health check implementation
- **EtcdServer**: etcd-specific health check implementation
- **ServerStatus**: Health check result struct

## Key Functions

- **NewStorage(serverRetriever)**: Creates REST with function to get servers to validate
- **List()**: Probes all servers concurrently and returns their health status
- **Get()**: Probes a specific server and returns its health status
- **ShortNames()**: Returns ["cs"] for kubectl
- **ToConditionStatus()**: Converts probe.Result to ConditionStatus
- **HttpServer.DoServerCheck()**: Performs HTTP health check with optional TLS and validation
- **EtcdServer.DoServerCheck()**: Performs etcd health check using storage prober

## Design Notes

- Read-only resource (no Create, Update, Delete)
- No backing etcd storage - health is probed live on each request
- Cluster-scoped resource
- List() probes all servers concurrently using goroutines
- Label/field selectors are supported but ComponentStatus doesn't support labels
- Uses 20-second probe timeout
- HttpServer supports custom validation functions for response body
- DEPRECATED: componentstatus is deprecated and will be removed
