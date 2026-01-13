# Package: hostutil

## Purpose
Provides utilities for interacting with the host filesystem, including file type detection, path operations, and SELinux support.

## Key Types/Structs
- `HostUtils` - Interface for host filesystem operations
- `HostUtil` - Linux implementation of HostUtils
- `FakeHostUtil` - Fake implementation for testing
- `FileType` - Enum for file types (Directory, File, Socket, BlockDevice, CharDevice)

## Key Functions
- `DeviceOpened()` - Checks if device is in use (mounted)
- `PathIsDevice()` - Determines if path is a device
- `MakeRShared()` - Ensures path has rshared mount propagation
- `GetFileType()` - Returns file type for path
- `PathExists()` - Checks if path exists
- `EvalHostSymlinks()` - Resolves symbolic links
- `GetSELinuxSupport()` - Checks if mount supports SELinux
- `GetSELinuxMountContext()` - Gets SELinux context of mount

## Design Patterns
- Platform-specific implementations (Linux, Windows, unsupported)
- Interface-based design for testability
- Centralizes host-specific operations for volume plugins
