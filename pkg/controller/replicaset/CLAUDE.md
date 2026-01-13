# Package: replicaset

Implements the ReplicaSet controller that maintains a stable set of replica Pods running at any given time.

## Key Types

- **ReplicaSetController**: Main controller that syncs ReplicaSets with pods.
- **expectations**: Tracks expected pod creations/deletions to avoid premature syncs.
- **RSControllerRefManager**: Manages pod ownership via ControllerRef.

## Key Functions

- **NewReplicaSetController**: Creates a new controller with informers and client.
- **Run**: Starts workers that process ReplicaSet reconciliation.
- **syncReplicaSet**: Main sync logic - reconciles desired vs actual replica count.
- **manageReplicas**: Creates or deletes pods to match desired count.
- **claimPods**: Adopts/releases pods based on selector and ownership.
- **getPodsToDelete**: Selects pods for deletion based on status and distribution.

## Key Features

- **Slow-start batch pattern**: Creates pods in exponentially growing batches (1, 2, 4, 8...) up to configured burst size.
- **Active pod tracking**: Only counts non-failed, non-deleted pods toward replica count.
- **Controller expectations**: Prevents rapid re-syncs by tracking anticipated changes.
- **Pod adoption/release**: Claims orphan pods matching selector; releases non-matching owned pods.

## Design Patterns

- Uses workqueue with rate-limited retries for processing.
- Implements the Kubernetes controller pattern with informers and listers.
- Supports both ReplicaSet and ReplicationController (via adapters).
- Prioritizes pod deletion by status (unknown > unhealthy > newer).
- Records events for important state transitions.
