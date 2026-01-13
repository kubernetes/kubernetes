# Package: probe

## Purpose
The `probe` package defines common types and utilities for container health probes (liveness, readiness, startup).

## Key Types

- **Result**: String type representing probe results.
  - `Success`: Probe succeeded.
  - `Warning`: Logically success with debugging info.
  - `Failure`: Probe failed.
  - `Unknown`: Probe result unknown.

## Key Functions

- **ProbeDialer**: Returns a net.Dialer configured for probing (platform-specific).

## Platform Support

- **Linux/Others**: Standard dialer implementation.
- **Windows**: Special dialer with Windows-specific settings.

## Subpackages

- **exec**: Command execution probe.
- **grpc**: gRPC health check probe.
- **http**: HTTP GET probe.
- **tcp**: TCP socket probe.

## Design Notes

- Result type provides consistent probe outcome representation.
- ProbeDialer abstracts platform-specific networking differences.
- Used by kubelet prober to check container health.
- Each probe type has its own Prober interface.
