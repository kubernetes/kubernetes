# Package: daemon

## Purpose
Implements the DaemonSet controller that ensures a copy of a Pod runs on all (or selected) nodes in the cluster.

## Key Types/Structs
- `DaemonSetsController`: Main controller with Pod/DaemonSet listers, ControllerRevision management, and workqueue

## Key Constants
- `BurstReplicas`: 250 - rate limit for pod creation to prevent registry DoS
- `StatusUpdateRetries`: 1 - retries for status updates
- `BackoffGCInterval`: 1 minute - interval for backoff garbage collection

## Event Reasons
- `SelectingAllReason`: DaemonSet selects all Pods
- `FailedPlacementReason`: Cannot schedule Pod to a node
- `FailedDaemonPodReason` / `SucceededDaemonPodReason`: Pod lifecycle events

## Key Functions
- `NewDaemonSetsController(dsInformer, historyInformer, podInformer, nodeInformer, kubeClient, failedPodsBackoff)`: Creates controller
- `Run(ctx, workers)`: Starts the controller
- `syncDaemonSet(ctx, key)`: Main sync loop for a single DaemonSet
- `manage(ctx, ds, nodeList, hash)`: Creates/deletes Pods to match desired state

## Node Selection
- Respects node selectors and affinity rules
- Handles taints and tolerations
- Supports rolling updates with maxSurge and maxUnavailable

## Design Notes
- Uses ControllerRevisions for rollback support
- Watches DaemonSets, Pods, Nodes, and ControllerRevisions
- Expectations tracking prevents duplicate pod creation
- Supports RollingUpdate and OnDelete update strategies
