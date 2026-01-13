# Package: nodevolumelimits

## Purpose
Implements a filter plugin that checks if attaching volumes to a node would exceed the node's volume attachment limits. Supports CSI volumes and handles in-tree to CSI migration.

## Key Types

### CSILimits
The plugin struct implementing:
- PreFilterPlugin, FilterPlugin, EnqueueExtensions, SignPlugin
- **csiManager**: CSI driver manager
- **pvLister/pvcLister/scLister**: Volume-related listers
- **vaLister**: VolumeAttachment lister
- **csiDriverLister**: CSI driver lister
- **translator**: In-tree to CSI translation

### InTreeToCSITranslator (interface)
Methods for handling in-tree volume migration:
- IsPVMigratable, IsInlineMigratable
- GetCSINameFromInTreeName
- TranslateInTreePVToCSI

## Extension Points

### PreFilter
- Counts volumes per CSI driver for the pod
- Handles both PVC-backed and inline volumes
- Caches volume counts in CycleState

### Filter
- Gets node's volume limits from CSINode object
- Compares current attachments + pod volumes against limits
- Returns Unschedulable if limits would be exceeded

## Key Functions

- **NewCSI(ctx, args, handle, features)**: Creates the CSI limits plugin
- **EventsToRegister()**: Returns CSINode, Pod/Delete, PVC/Add, VolumeAttachment/Delete events
- **getVolumeLimits(csiNode, node)**: Extracts per-driver attachment limits

## Volume Counting Logic
1. Count volumes by CSI driver name
2. Handle in-tree volumes via CSI migration
3. Skip volumes that don't count against limits (e.g., local volumes)
4. Account for volumes already attached to the node

## Design Pattern
- Supports both CSI-native and migrated in-tree volumes
- Uses random prefix for volume IDs to handle unnamed ephemeral volumes
- Volume limit scaling support via feature gate
- Queueing hints for volume-related events
