# Package: recyclerclient

## Purpose
Provides a client for managing recycler pods that clean persistent volume data before reuse.

## Key Types/Structs
- `recyclerClient` - Interface for pod operations during recycling
- `realRecyclerClient` - Implementation using Kubernetes client
- `RecycleEventRecorder` - Function type for recording recycle events

## Key Functions
- `RecycleVolumeByWatchingPodUntilCompletion()` - Main entry point for volume recycling
- `internalRecycleVolumeByWatchingPodUntilCompletion()` - Creates and monitors recycler pod
- `waitForPod()` - Watches pod until completion or failure
- `WatchPod()` - Sets up watch for pod and its events

## Design Patterns
- Creates a pod to execute volume cleanup (e.g., rm -rf)
- Watches pod until completion, failure, or timeout
- Handles existing recycler pods by deleting them
- Forwards pod events to PV for visibility
- Ensures recycler pod cleanup even on errors
- Designed for testability with interface abstraction
