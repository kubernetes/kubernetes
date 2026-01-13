# Package: subpath

## Purpose
Provides secure subpath handling for volume mounts, ensuring containers can only access designated subdirectories within volumes.

## Key Types/Structs
- `Interface` - Subpath operations interface
- `Subpath` - Describes a subpath mount (path, volume, pod, container info)
- `subpath` - Linux implementation
- `FakeSubpath` - Fake implementation for testing

## Key Functions
- `PrepareSafeSubpath()` - Prepares immutable subpath for container use
- `CleanSubPaths()` - Removes bind mounts created by PrepareSafeSubpath
- `SafeMakeDir()` - Creates directory safely without following symlinks

## Design Patterns
- Prevents symlink attacks where container escapes volume boundary
- Uses bind mounts to lock subpath location at container start
- Platform-specific implementations (Linux, Windows, unsupported)
- Ensures subpath cannot be modified by container to escape volume
- Returns cleanup action to be called after container starts
