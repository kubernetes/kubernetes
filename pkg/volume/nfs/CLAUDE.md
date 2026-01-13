# Package: nfs

## Purpose
Implements the NFS volume plugin for mounting NFS shares as volumes in pods.

## Key Types/Structs
- `nfsPlugin` - VolumePlugin for NFS volumes
- `nfsMounter` - Handles NFS volume mounting
- `nfsUnmounter` - Handles NFS volume unmounting

## Key Functions
- `ProbeVolumePlugins()` - Returns the NFS plugin
- `SetUpAt()` - Mounts NFS share to pod volume directory
- `TearDownAt()` - Unmounts NFS share
- `GetAccessModes()` - Returns ReadWriteOnce, ReadOnlyMany, ReadWriteMany

## Design Patterns
- Simple network filesystem mount using system mount command
- Supports mount options via annotations or PV spec
- Supports all access modes including ReadWriteMany
- No attach/detach phase (directly mounts to pod)
- Implements RecyclableVolumePlugin for PV recycling
