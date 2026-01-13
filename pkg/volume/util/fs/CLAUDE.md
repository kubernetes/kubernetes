# Package: fs

## Purpose
Provides filesystem utility functions for calculating disk usage, available space, and inode statistics.

## Key Types/Structs
- `UsageInfo` - Contains Bytes and Inodes usage counts

## Key Functions
- `Info()` - Returns filesystem stats: available, capacity, usage, inodes, inodes free
- `DiskUsage()` - Calculates disk usage for a directory (bytes and inodes)

## Design Patterns
- Platform-specific implementations (Linux/Darwin vs Windows vs unsupported)
- Uses statfs syscall for filesystem information on Unix
- DiskUsage walks directory tree counting blocks and inodes
- Integrates with fsquota when available for faster quota-based usage
- Handles hardlinks by deduplicating inode counts
- Skips directories on different devices to avoid crossing mount boundaries
