# Package: local

## Purpose
Implements the Local persistent volume plugin that uses local storage devices or directories on specific nodes.

## Key Types/Structs
- `localVolumePlugin` - VolumePlugin for local volumes
- `localVolumeMounter` - Handles local volume mounting
- `localVolumeUnmounter` - Handles local volume unmounting
- `localVolumeMapper` - Block volume mapper for raw device access

## Key Functions
- `ProbeVolumePlugins()` - Returns the local volume plugin
- `SetUpAt()` - Bind mounts local path to pod volume directory
- `TearDownAt()` - Unmounts the bind mount
- `CanDeviceMount()` - Checks if path is a block device for device mounting

## Design Patterns
- Requires node affinity for scheduling to correct node
- Supports both filesystem paths and block devices
- No automatic provisioning (admin must pre-create PVs)
- Uses bind mounts for filesystem volumes
- Does not support dynamic provisioning (volumeBindingMode: WaitForFirstConsumer recommended)
