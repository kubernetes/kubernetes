# Package: status

## Purpose
The `status` package manages pod status updates for the Kubelet. It serves as the source of truth for pod status on the node and synchronizes status changes to the API server.

## Key Types/Structs

- **Manager**: Interface for managing pod status. Provides methods to get/set pod status, container readiness/startup, and handle pod termination.
- **manager**: Implementation that caches pod statuses and syncs with the API server periodically.
- **versionedPodStatus**: Wrapper around `v1.PodStatus` with version tracking to prevent stale updates.
- **PodStatusProvider**: Interface for getting cached pod status by UID.
- **PodManager**: Interface subset for pod lookups and UID translations (static pods to mirror pods).
- **PodDeletionSafetyProvider**: Interface to check if a pod could have running containers before deletion.
- **podResizeConditions**: Tracks in-progress and pending resize conditions for in-place pod vertical scaling.

## Key Functions

- **NewManager**: Creates a new status manager.
- **Start**: Begins the background sync loop that updates the API server.
- **SetPodStatus**: Caches status and triggers sync.
- **SetContainerReadiness/SetContainerStartup**: Updates container-level status.
- **TerminatePod**: Ensures containers have terminal state at pod end-of-life.
- **RemoveOrphanedStatuses**: Cleans up statuses for deleted pods.
- **GetPodStatus**: Returns cached status for a pod.
- **syncBatch**: Syncs pending status updates to API server.
- **mergePodStatus**: Preserves non-kubelet-owned conditions when updating status.

## Design Notes

- Uses versioned status to ensure monotonic progress and prevent clobbering concurrent updates.
- Syncs to API server every 10 seconds (syncPeriod) or immediately on status channel signal.
- Handles static pods by translating between pod UID and mirror pod UID.
- Enforces illegal state transition detection (terminated -> non-terminated).
- Supports in-place pod vertical scaling via PodResizePending/PodResizeInProgress conditions.
- Terminal pod phases (Succeeded/Failed) are only set after all containers terminate.
- Pod deletion from API server only happens after pod is fully terminated.
