# Package: portworx

## Purpose
Implements the Portworx volume plugin for mounting Portworx distributed storage volumes.

## Key Types/Structs
- `portworxVolumePlugin` - VolumePlugin for Portworx volumes
- `portworxVolumeMounter` - Handles Portworx volume mounting
- `portworxVolumeUnmounter` - Handles Portworx volume unmounting
- `portworxVolumeProvisioner` - Dynamic provisioner for Portworx volumes

## Key Functions
- `ProbeVolumePlugins()` - Returns the Portworx plugin
- `SetUpAt()` - Attaches and mounts Portworx volume
- `TearDownAt()` - Unmounts and detaches Portworx volume
- `Provision()` - Creates new Portworx volume

## Design Patterns
- Communicates with Portworx daemon via local gRPC client
- Supports dynamic provisioning with storage class parameters
- Supports volume expansion
- Implements AttachableVolumePlugin interface
- Uses Portworx-specific API for volume operations
