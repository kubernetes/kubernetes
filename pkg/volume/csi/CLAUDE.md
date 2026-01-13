# Package: csi

## Purpose
Implements the Container Storage Interface (CSI) volume plugin, enabling Kubernetes to use external CSI drivers for storage provisioning and management.

## Key Types/Structs
- `csiPlugin` - Main plugin implementing multiple volume interfaces
- `csiMountMgr` - Handles CSI NodePublishVolume/NodeUnpublishVolume for filesystem mounts
- `csiAttacher` - Handles CSI ControllerPublish and VolumeAttachment management
- `csiDriverClient` - gRPC client wrapper for CSI driver communication
- `RegistrationHandler` - Handles CSI driver registration via kubelet plugin watcher

## Key Functions
- `ProbeVolumePlugins()` - Returns the CSI plugin
- `NewMounter/NewUnmounter` - Create CSI volume mounters
- `NewAttacher/NewDetacher` - Create CSI volume attachers (for controller)
- `NewBlockVolumeMapper` - Create block volume mapper for raw block access
- `RegistrationHandler.RegisterPlugin` - Register CSI driver with kubelet

## Design Patterns
- gRPC communication with external CSI drivers via unix socket
- VolumeAttachment CRD for tracking attach/detach state
- Supports both filesystem and block volume modes
- CSI migration from in-tree plugins to CSI drivers
- SELinux mount context support via NodePublishVolume
