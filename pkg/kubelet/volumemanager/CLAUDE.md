# Package: volumemanager

## Purpose
The `volumemanager` package manages volume attachment, mounting, and unmounting for pods on a kubelet node.

## Key Interfaces

- **VolumeManager**: Main interface for managing volumes.
  - `Run()`: Starts the volume manager loops.
  - `WaitForAttachAndMount()`: Blocks until volumes are attached/mounted for a pod.
  - `WaitForUnmount()`: Blocks until volumes are unmounted for a pod.
  - `GetMountedVolumesForPod()`: Returns currently mounted volumes for a pod.
  - `GetVolumesInUse()`: Returns all volumes in use on this node.
  - `MarkVolumesAsReportedInUse()`: Marks volumes as reported to API server.

## Key Types

- **volumeManager**: Implements VolumeManager with DSW (Desired State of World), ASW (Actual State of World), reconciler, and populator components.

## Architecture

1. **Desired State of World (DSW)**: Tracks which volumes should be attached/mounted.
2. **Actual State of World (ASW)**: Tracks which volumes are currently attached/mounted.
3. **Reconciler**: Drives ASW toward DSW by triggering attach/mount/unmount/detach operations.
4. **Populator**: Monitors pods and updates DSW based on pod volume requirements.

## Design Notes

- Uses a reconciliation loop pattern (desired vs actual state).
- Supports SELinux mount context labels for volumes.
- Integrates with CSI and in-tree volume plugins.
- Coordinates with attach/detach controller for multi-attach scenarios.
