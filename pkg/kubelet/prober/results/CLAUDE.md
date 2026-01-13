# Package: results

Provides probe result caching and update notification for the prober.

## Key Types

- **Manager**: Interface for caching and watching probe results.
- **manager**: Implementation with map-based cache and update channel.
- **Result**: Enum type for probe outcomes (Success=0, Failure=1, Unknown=-1).
- **Update**: Notification struct with ContainerID, Result, and PodUID.

## Key Functions

- `NewManager()`: Creates an empty results manager with buffered update channel.
- `Get(id)`: Returns cached result for container, or (Unknown, false) if not found.
- `Set(id, result, pod)`: Caches result and sends update if result changed.
- `Remove(id)`: Removes cached result for container.
- `Updates()`: Returns channel for receiving result change notifications.

## Result Values

- `Unknown (-1)`: Initial state, probe not yet run
- `Success (0)`: Probe passed
- `Failure (1)`: Probe failed

## Design Notes

- Update channel has buffer size of 20
- Only sends updates when result actually changes (not on duplicate sets)
- Currently supports single subscriber (one Updates() call)
- Thread-safe with RWMutex protection
- Result.ToPrometheusType() converts to float64 for metrics
