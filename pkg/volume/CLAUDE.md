# Package: volume

## Purpose
Core volume plugin framework for Kubernetes. Defines interfaces and types for implementing volume plugins that handle mounting, unmounting, attaching, and detaching volumes to pods and nodes.

## Key Types/Structs
- `Volume` - Base interface with GetPath() and MetricsProvider
- `Mounter/Unmounter` - Interfaces for mounting/unmounting volumes to pods
- `Attacher/Detacher` - Interfaces for attaching/detaching volumes to nodes
- `BlockVolume/BlockVolumeMapper` - Interfaces for raw block device volumes
- `Provisioner/Deleter` - Interfaces for dynamic volume provisioning
- `DeviceMounter/DeviceUnmounter` - Interfaces for global device mounts
- `Metrics` - Volume usage statistics (capacity, used, available, inodes)
- `Spec` - Internal volume representation wrapping v1.Volume or v1.PersistentVolume
- `VolumePluginMgr` - Plugin registry managing all volume plugins

## Key Functions
- `NewSpecFromVolume/NewSpecFromPersistentVolume` - Create Spec from API types
- `NewMetricsDu/NewMetricsStatFS/NewMetricsBlock` - Create metrics providers
- `VolumePluginMgr.InitPlugins` - Initialize and register volume plugins
- `VolumePluginMgr.FindPluginBySpec/FindPluginByName` - Plugin lookup methods

## Design Patterns
- Plugin architecture with interface-based extensibility
- Idempotent operations (mount/unmount can be called multiple times safely)
- Separation of filesystem volumes vs block volumes
- Two-phase attach: global device mount, then per-pod bind mount
