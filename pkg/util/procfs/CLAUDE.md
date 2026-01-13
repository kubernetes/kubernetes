# Package: procfs

## Purpose
Provides utilities for working with the Linux /proc filesystem, including process discovery and container identification.

## Key Types
- `ProcFSInterface` - Interface for procfs operations
- `ProcFS` - Linux implementation of ProcFSInterface

## Key Functions
- `GetFullContainerName()` - Gets container name from process's cgroup membership
- `PidOf()` - Finds PIDs of processes matching a name pattern
- `PKill()` - Sends a signal to processes matching a name pattern

## Implementation Details
- Reads /proc/<pid>/cgroup to determine container membership
- Reads /proc/<pid>/cmdline to match process names
- Uses regexp matching for flexible process identification
- Handles process churn during iteration

## Platform Support
- Linux: Full implementation reading /proc
- Other platforms: Stub/fake implementations

## Design Patterns
- Interface-based for testability
- Streaming directory reads for large process counts
- Signal sending with aggregated error reporting
