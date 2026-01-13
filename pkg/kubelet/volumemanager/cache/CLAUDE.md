# Package: cache

## Purpose
The `cache` package provides Desired State of World (DSW) and Actual State of World (ASW) caches for tracking volume state in the kubelet volume manager.

## Key Interfaces

### DesiredStateOfWorld (DSW)
- `AddPodToVolume()`: Marks a volume as needing to be mounted for a pod.
- `DeletePodFromVolume()`: Removes a pod's requirement for a volume.
- `MarkVolumesReportedInUse()`: Updates which volumes are reported in use to API server.
- `GetVolumesToMount()`: Returns all volumes that should be attached/mounted.
- `GetPods()`: Returns pods that have volumes in DSW.

### ActualStateOfWorld (ASW)
- `MarkVolumeAsAttached()`: Records that a volume is attached to the node.
- `MarkVolumeAsMounted()`: Records that a volume is mounted for a pod.
- `MarkVolumeAsUnmounted()`: Records that a volume was unmounted.
- `MarkVolumeAsDetached()`: Records that a volume was detached.
- `GetMountedVolumes()`: Returns all currently mounted volumes.
- `GetAttachedVolumes()`: Returns all currently attached volumes.

## Key Types

- **MountedVolume**: Contains information about a mounted volume (pod, plugin, paths, SELinux context).
- **AttachedVolume**: Contains information about an attached volume (plugin name, device path).

## Design Notes

- Thread-safe via mutex protection.
- ASW tracks both attachment state and mount state separately.
- Supports SELinux context tracking for mounts.
- Handles uncertain attachment states during kubelet restart.
- Volume reconstruction recovers state after kubelet crashes.
