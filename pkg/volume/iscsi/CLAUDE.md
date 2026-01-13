# Package: iscsi

## Purpose
Implements the iSCSI volume plugin for mounting iSCSI LUNs as volumes in pods.

## Key Types/Structs
- `iscsiPlugin` - VolumePlugin for iSCSI volumes
- `iscsiDiskMounter` - Handles iSCSI disk mounting
- `iscsiDiskUnmounter` - Handles iSCSI disk unmounting
- `iscsiDiskMapper` - Block volume mapper for raw iSCSI device access

## Key Functions
- `ProbeVolumePlugins()` - Returns the iSCSI plugin
- `SetUpAt()` - Discovers and mounts iSCSI target
- `TearDownAt()` - Unmounts and logs out from iSCSI target
- `AttachDisk()` - Performs iSCSI login and device discovery

## Design Patterns
- Supports CHAP authentication (initiator and target)
- Uses iscsiadm for target discovery and login
- Supports multipath for high availability
- Supports both filesystem and block volume modes
- Global device mount followed by bind mount to pod
