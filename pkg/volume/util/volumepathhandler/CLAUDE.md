# Package: volumepathhandler

## Purpose
Handles block volume device path operations including mapping devices to pods via symbolic links or bind mounts.

## Key Types/Structs
- `BlockVolumePathHandler` - Interface for block volume path operations
- `VolumePathHandler` - Implementation of block volume path operations

## Key Functions
- `MapDevice()` - Creates symbolic link or bind mount to block device
- `UnmapDevice()` - Removes symbolic link or bind mount
- `RemoveMapPath()` - Removes map path directory
- `IsSymlinkExist()` - Checks if symbolic link exists
- `IsDeviceBindMountExist()` - Checks if bind mount exists
- `GetDeviceBindMountRefs()` - Finds bind mount references under path
- `FindGlobalMapPathUUIDFromPod()` - Locates global map path for pod
- `AttachFileDevice()` - Attaches file as loop device
- `DetachFileDevice()` - Detaches loop device
- `GetLoopDevice()` - Gets loop device for file path

## Design Patterns
- Two mapping modes: symbolic links (default) or bind mounts
- Global map path for device, pod map path for per-pod access
- Loop device support for file-backed block volumes
- Platform-specific: full support on Linux, stubs elsewhere
- Uses losetup for loop device management
