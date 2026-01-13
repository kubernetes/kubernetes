# Package: statusupdater

## Purpose
Implements interfaces for updating the VolumesAttached field in Node Status API objects based on the actual state of world cache.

## Key Types

- **NodeStatusUpdater**: Interface for updating node volume attachment status.
- **nodeStatusUpdater**: Implementation that patches Node.Status.VolumesAttached.

## Key Functions

- **NewNodeStatusUpdater(kubeClient, nodeLister, actualStateOfWorld)**: Creates a new updater.
- **UpdateNodeStatuses(logger)**: Updates all nodes that have pending status changes.
- **UpdateNodeStatusForNode(logger, nodeName)**: Updates status for a specific node.
- **processNodeVolumes**: Retrieves node and applies status update.
- **updateNodeStatus**: Performs the actual patch operation using nodeutil.PatchNodeStatus.

## Design Notes

- Uses `GetVolumesToReportAttached` from ASW to determine which nodes need updates.
- Only updates nodes where ASW indicates statusUpdateNeeded is true.
- On update failure, re-marks the node for update via SetNodeStatusUpdateNeeded.
- Silently ignores NotFound errors (node deleted).
- Deep copies node object before modification to avoid cache mutation.
- Uses strategic merge patch for efficient updates.
