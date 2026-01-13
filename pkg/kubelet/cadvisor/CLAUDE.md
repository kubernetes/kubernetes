# Package cadvisor

Package cadvisor provides the kubelet's interface to cAdvisor for collecting container and machine resource metrics.

## Key Types

- `Interface`: Abstract interface to cAdvisor for container stats, machine info, and filesystem info
- `ImageFsInfoProvider`: Interface for determining the filesystem labels for container images
- `cadvisorClient`: Linux implementation wrapping the cAdvisor manager

## Key Functions

- `New`: Creates a new cAdvisor client on Linux with configurable metrics collection
- `Start`: Starts the cAdvisor manager
- `MachineInfo`: Returns machine hardware information (cores, memory, etc.)
- `ContainerInfoV2`: Returns container resource usage statistics
- `ImagesFsInfo`: Returns filesystem info for container images
- `RootFsInfo`: Returns filesystem info for the root filesystem
- `ContainerFsInfo`: Returns filesystem info for container writable layer

## Platform Support

- `cadvisor_linux.go`: Full cAdvisor integration with all container runtimes
- `cadvisor_windows.go`: Windows-specific implementation
- `cadvisor_unsupported.go`: Stub for unsupported platforms

## Design Notes

- Registers container handlers for containerd, CRI-O, and systemd
- Registers filesystem plugins for btrfs, devicemapper, overlay, tmpfs, etc.
- Configurable housekeeping interval (default 10s, max 15s)
- Stats cached for 2 minutes
- PSI metrics enabled via KubeletPSI feature gate
