# Package: statefulset

Implements the StatefulSet controller that manages stateful applications with stable network identities and persistent storage.

## Key Types

- **StatefulSetController**: Main controller syncing StatefulSets with pods.
- **StatefulSetControlInterface**: Interface abstracting StatefulSet update operations.
- **StatefulPodControl**: Handles pod operations specific to StatefulSets.
- **StatefulSetStatusUpdater**: Updates StatefulSet status subresource.

## Key Functions

- **NewStatefulSetController**: Creates controller with informers for pods, sets, PVCs, and revisions.
- **Run**: Starts workers processing StatefulSet reconciliation.
- **sync**: Main sync logic - retrieves pods and delegates to syncStatefulSet.
- **syncStatefulSet**: Coordinates pod control and status updates.
- **getPodsForStatefulSet**: Returns pods owned by the StatefulSet.
- **adoptOrphanRevisions**: Adopts unowned ControllerRevisions matching the set.

## Key Features

- Ordered pod management (create/delete in ordinal sequence).
- Stable network identity via headless service.
- Persistent storage via PVC templates.
- Rolling updates with partition support.
- MinReadySeconds for availability tracking.
- Controller revision history for rollback.

## Design Patterns

- Uses ControllerRevisions for update history and rollback.
- Supports both OrderedReady and Parallel pod management policies.
- Integrates with history package for revision management.
- Exposes metrics for max unavailable and actual unavailable pods.
- Handles pod adoption/release via ControllerRefManager.
