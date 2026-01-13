# Package: tcp

## Purpose
The `tcp` package implements TCP socket probes for container health checks.

## Key Interfaces

- **Prober**: Interface for TCP probes.
  - `Probe(host, port, timeout)`: Attempts TCP connection.

## Key Functions

- **New**: Creates a new TCP Prober.
- **DoTCPProbe**: Performs TCP probe (exported for direct use).

## Behavior

1. Attempts to open TCP connection to host:port.
2. Uses configured timeout for connection attempt.
3. Returns Success if connection opens successfully.
4. Returns Failure if connection fails or times out.
5. Immediately closes connection after successful open.

## Design Notes

- Simplest probe type - only checks if port is accepting connections.
- Uses ProbeDialer for platform-specific networking.
- Connection errors converted to Failure (not Unknown).
- No data is sent or received - just connection test.
- Used by kubelet for TCP-based liveness/readiness probes.
