# Package: removeall

## Purpose
Provides filesystem removal utilities that respect mount boundaries, preventing accidental deletion of files on mounted filesystems.

## Key Functions
- `RemoveAllOneFilesystem()` - Removes path and children without crossing mount boundaries
- `RemoveDirsOneFilesystem()` - Removes only empty directories, fails if files exist
- `RemoveAllOneFilesystemCommon()` - Common implementation with pluggable remove function

## Behavior
- Similar to `rm -rf --one-file-system`
- Returns error if path is a mount point
- Recursively removes contents before directory
- Returns nil if path doesn't exist

## Design Patterns
- Uses mount.Interface to detect mount points
- Pluggable remove function (os.Remove vs syscall.Rmdir)
- Handles race conditions between stat and remove
- Used by volume cleanup to avoid removing mounted volumes
