# Package: cache

## Purpose
Implements thread-safe data structures for tracking volume attachment state in the attach/detach controller: the Actual State of World (ASW) and Desired State of World (DSW).

## Key Types

### ActualStateOfWorld
- **ActualStateOfWorld**: Interface for tracking which volumes are attached to which nodes.
- **AttachedVolume**: Represents a volume attached to a node with mount status and detach timing.
- **AttachState**: Enum (AttachStateAttached, AttachStateUncertain, AttachStateDetached).

### DesiredStateOfWorld
- **DesiredStateOfWorld**: Interface for tracking desired volume attachments (nodes->volumes->pods).
- **VolumeToAttach**: Represents a volume that should be attached to a node.
- **PodToAdd**: Represents a pod referencing a volume scheduled to a node.

## Key Functions

### ActualStateOfWorld
- **NewActualStateOfWorld(pluginMgr)**: Creates new ASW instance.
- **MarkVolumeAsAttached/MarkVolumeAsDetached/MarkVolumeAsUncertain**: Update attachment state.
- **SetVolumesMountedByNode**: Track which volumes are in use (unsafe to detach).
- **GetAttachState**: Query attachment state for volume/node pair.
- **GetVolumesToReportAttached**: Get volumes to report in Node.Status.VolumesAttached.

### DesiredStateOfWorld
- **NewDesiredStateOfWorld(pluginMgr)**: Creates new DSW instance.
- **AddNode/DeleteNode**: Manage nodes in DSW.
- **AddPod/DeletePod**: Manage pod-volume-node associations.
- **GetVolumesToAttach**: List all volumes needing attachment.
- **SetMultiAttachError**: Mark multi-attach errors as reported.

## Design Notes

- Both caches are distinct from kubelet's volume manager caches (different purposes).
- ASW tracks node status updates needed for Node API object.
- DSW prevents node deletion while volumes are attached.
- Thread-safe via sync.RWMutex.
