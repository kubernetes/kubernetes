# Package: prober

Manages container liveness, readiness, and startup probes.

## Key Types

- **Manager**: Interface for managing probe workers and updating pod status.
- **manager**: Implementation coordinating probe workers and result managers.
- **worker**: Executes individual probes for a container at configured intervals.
- **prober**: Performs actual probe execution (exec, HTTP, TCP, gRPC).
- **probeKey**: Unique identifier for a probe (podUID + containerName + probeType).

## Key Functions

- `NewManager()`: Creates probe manager with status and results managers.
- `AddPod()`: Creates probe workers for all container probes in a pod.
- `StopLivenessAndStartup()`: Stops liveness/startup probes during termination.
- `RemovePod()`: Cleans up probe workers and cached results.
- `UpdatePodStatus()`: Sets container Ready state based on probe results.

## Probe Types

- **Liveness**: Restarts container if probe fails
- **Readiness**: Removes container from service endpoints if probe fails
- **Startup**: Blocks other probes until container is ready

## Probe Methods (in prober.go)

- **Exec**: Runs command in container
- **HTTPGet**: Makes HTTP request to container
- **TCPSocket**: Opens TCP connection to container
- **gRPC**: Makes gRPC health check call

## Metrics

- `prober_probe_total`: Cumulative probe count by type, result, container (BETA)
- `prober_probe_duration_seconds`: Probe response duration histogram (ALPHA)

## Design Notes

- Each container probe gets its own worker goroutine
- Workers cache results; manager reads cache for status updates
- Supports initial delay, period, timeout, success/failure thresholds
- HTTP probes support configurable headers and redirect following
