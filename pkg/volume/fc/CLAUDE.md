# Package: fc

## Purpose
Implements the Fibre Channel (FC) volume plugin for mounting FC/FCoE LUNs as volumes in pods.

## Key Types/Structs
- `fcPlugin` - VolumePlugin for FC volumes
- `fcDiskMounter` - Handles FC volume mounting
- `fcDiskUnmounter` - Handles FC volume unmounting
- `fcDiskMapper` - Block volume mapper for raw FC device access

## Key Functions
- `ProbeVolumePlugins()` - Returns the FC plugin
- `SetUpAt()` - Mounts FC disk to pod directory
- `TearDownAt()` - Unmounts FC disk
- `AttachDisk()` - Attaches FC disk using SCSI discovery

## Design Patterns
- Supports both WWID and WWNN/WWPN targeting
- Uses multipath for high availability configurations
- Supports both filesystem and block volume modes
- Global mount followed by bind mount to pod directory
- Implements PersistentVolumePlugin interface
