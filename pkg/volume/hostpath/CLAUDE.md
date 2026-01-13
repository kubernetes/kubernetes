# Package: hostpath

## Purpose
Implements the HostPath volume plugin that mounts a file or directory from the host node's filesystem into a pod.

## Key Types/Structs
- `hostPathPlugin` - VolumePlugin for hostpath volumes
- `hostPathVolume` - Mounter/Unmounter implementation
- `hostPathProvisioner` - Dynamic provisioner for hostpath PVs (testing only)
- `hostPathDeleter` - Deletes hostpath PV data

## Key Functions
- `ProbeVolumePlugins()` - Returns the hostpath plugin
- `SetUpAt()` - Bind mounts host path to pod volume directory
- `TearDownAt()` - Unmounts the bind mount
- `checkType()` - Validates host path type (File, Directory, Socket, etc.)

## Design Patterns
- Supports path type validation: Directory, File, Socket, CharDevice, BlockDevice
- DirectoryOrCreate/FileOrCreate types auto-create missing paths
- Implements RecyclableVolumePlugin for PV recycling
- WARNING: Security risk - provides direct host filesystem access
- Primarily for single-node testing, not production multi-node clusters
