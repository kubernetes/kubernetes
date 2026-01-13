# Package: nodestatus

Provides setter functions for updating Kubernetes Node status fields.

## Key Types

- **Setter**: Function type `func(ctx, node) error` that modifies a Node object in-place.

## Setter Functions

- **NodeAddress()**: Sets node addresses (InternalIP, Hostname) based on nodeIPs configuration. Handles single/dual-stack, external cloud providers, and address validation.
- **MachineInfo()**: Sets machine information (OS, architecture, boot ID, kernel version, OS image, container runtime version).
- **VersionInfo()**: Sets kubelet and kube-proxy version in node status.
- **DaemonEndpoints()**: Sets kubelet endpoint port.
- **Images()**: Sets list of container images on the node (limited to MaxNamesPerImageInNodeStatus=5 names per image).
- **ReadyCondition()**: Sets Ready condition based on runtime status, PLEG health, etc.
- **MemoryPressureCondition() / DiskPressureCondition() / PIDPressureCondition()**: Set pressure conditions from eviction manager.
- **VolumesInUse()**: Sets list of volumes currently in use.
- **VolumeLimit()**: Sets volume attachment limits per CSI driver.

## Design Notes

- Setters can partially mutate node before returning errors
- Boot ID change triggers a one-time NodeRebooted event
- External cloud providers use annotations to hint IP addresses
- Image list sorted by total size, with name deduplication
- Pressure conditions track last transition time separately
