# Package: grpc

## Purpose
The `grpc` package implements gRPC health check probes for container health checks using the standard gRPC health protocol.

## Key Interfaces

- **Prober**: Interface for gRPC probes.
  - `Probe(host, service, port, timeout)`: Performs gRPC health check.

## Key Functions

- **New**: Creates a new gRPC Prober.

## Behavior

1. Connects to the specified host:port with timeout.
2. Sends gRPC health check request (grpc.health.v1.Health).
3. Returns Success if response is SERVING.
4. Returns Failure for non-SERVING status or errors.
5. Handles connection timeout and RPC timeout separately.

## Protocol

- Uses standard gRPC health protocol: `grpc.health.v1.Health/Check`.
- Supports service-specific health checks.
- User agent: `kube-probe/<version>`.

## Design Notes

- Uses insecure credentials (TLS not currently supported for probes).
- Mimics grpc_health_probe tool behavior.
- Errors are always nil (failures returned as probe.Failure).
- Timeout applies to both connection and RPC call.
- Used by kubelet for gRPC-based liveness/readiness/startup probes.
