# Package util

Package util provides cgroup utility functions for the container manager.

## Key Functions

- `GetPids(cgroupPath)`: Returns process IDs in a cgroup (used for cgroup accounting)

## Constants

- `CgroupRoot`: "/sys/fs/cgroup" - base path for cgroup filesystem

## Implementation Details

GetPids handles both cgroup v1 and v2:
- v2: Directly uses cgroupPath relative to CgroupRoot
- v1: Resolves path through devices subsystem mountpoint

Helper functions (v1):
- `getCgroupV1Path`: Resolves full filesystem path for cgroup
- `getCgroupV1ParentPath`: Gets parent cgroup path for relative resolution

## Platform Support

- `cgroups_linux.go`: Full Linux implementation
- `cgroups_unsupported.go`: Stub for non-Linux platforms

## Design Notes

- Forked from opencontainers/runc libcontainer
- Handles nested container scenarios via GetOwnCgroup
- Uses libcontainercgroups for cross-version compatibility
