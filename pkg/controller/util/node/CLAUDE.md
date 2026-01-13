# Package: node

## Purpose
Provides utility functions for node-related controller operations, including pod deletion, status updates, event recording, and taint management.

## Key Functions

### Pod Management
- **DeletePods(ctx, kubeClient, pods, recorder, nodeName, nodeUID, daemonStore)**: Deletes all pods from a node, skipping DaemonSet-managed pods. Returns whether any pods remain pending deletion.
- **SetPodTerminationReason(ctx, kubeClient, pod, nodeName)**: Sets the "NodeUnreachable" reason and message on a pod's status.
- **MarkPodsNotReady(ctx, kubeClient, recorder, pods, nodeName)**: Updates the Ready condition of pods to False when their node becomes unavailable.

### Event Recording
- **RecordNodeEvent(ctx, recorder, nodeName, nodeUID, eventtype, reason, event)**: Records an event for a node.
- **RecordNodeStatusChange(logger, recorder, node, newStatus)**: Records a node status change event.

### Node Modifications
- **SwapNodeControllerTaint(ctx, kubeClient, taintsToAdd, taintsToRemove, node)**: Adds and removes taints from a node atomically.
- **AddOrUpdateLabelsOnNode(ctx, kubeClient, labelsToUpdate, node)**: Updates labels on a node.

### Event Handler Factories
- **CreateAddNodeHandler(f)**: Creates an informer add handler that deep copies the node.
- **CreateUpdateNodeHandler(f)**: Creates an informer update handler with deep copied nodes.
- **CreateDeleteNodeHandler(logger, f)**: Creates a delete handler that handles DeletedFinalStateUnknown tombstones.

### Condition Utilities
- **GetNodeCondition(status, conditionType)**: Extracts a specific condition from node status.

## Design Notes

- All functions deep copy objects before modification to avoid mutating shared cache objects.
- Handles DeletedFinalStateUnknown tombstones from informer caches gracefully.
- Used by node lifecycle controller, IPAM controller, and other node-related controllers.
