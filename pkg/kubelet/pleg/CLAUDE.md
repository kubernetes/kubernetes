# Package: pleg

Pod Lifecycle Event Generator (PLEG) - detects pod/container state changes and generates events.

## Key Types

- **PodLifecycleEvent**: Event reflecting pod state change with ID, Type, and Data.
- **PodLifeCycleEventType**: Event types (ContainerStarted, ContainerDied, ContainerRemoved, PodSync, ContainerChanged, ConditionMet).
- **PodLifecycleEventGenerator**: Interface for generating and watching pod lifecycle events.
- **GenericPLEG**: Polling-based implementation that periodically lists containers to detect changes.
- **RelistDuration**: Configuration for relist period and health threshold.
- **WatchCondition**: Function type for custom pod watch conditions.

## Key Functions

- `NewGenericPLEG()`: Creates a polling-based PLEG with specified relist duration and cache.
- `Watch()`: Returns channel for receiving PodLifecycleEvents.
- `Healthy()`: Returns whether PLEG is healthy (relisting within threshold).
- `SetPodWatchCondition()`: Sets a custom condition to watch for on a pod.
- `RunningContainerWatchCondition()`: Helper to create container-specific watch conditions.

## GenericPLEG Design

- Periodically lists all pods/containers from runtime
- Compares current state with previous state (podRecords)
- Generates events for state transitions
- Updates container cache with latest status
- Handles pods that failed inspection via podsToReinspect

## State Transitions

- Running -> Exited: ContainerDied
- Exited -> NonExistent: ContainerRemoved
- NonExistent -> Running: ContainerStarted
- Unknown state changes: ContainerChanged
