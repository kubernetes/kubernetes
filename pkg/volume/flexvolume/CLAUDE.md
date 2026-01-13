# Package: flexvolume

## Purpose
Implements the FlexVolume plugin interface that allows third-party volume drivers to be implemented as executable scripts/binaries on the node.

## Key Types/Structs
- `flexVolumePlugin` - Plugin that wraps external driver executables
- `flexVolumeMounter` - Mounter that calls external driver mount
- `flexVolumeUnmounter` - Unmounter that calls external driver unmount
- `flexVolumeAttacher` - Attacher for attachable flex volumes

## Key Functions
- `ProbeVolumePlugins()` - Discovers and loads flex volume drivers from disk
- `NewDriverCall()` - Creates a call to the external driver executable
- `SetUpAt()` - Invokes driver mount command
- `TearDownAt()` - Invokes driver unmount command

## Design Patterns
- External driver binaries in /usr/libexec/kubernetes/kubelet-plugins/volume/exec/
- JSON-based communication with driver executables
- Supports init, attach, detach, mount, unmount operations
- Driver capabilities discovery via init command
- Deprecated in favor of CSI, but still supported
