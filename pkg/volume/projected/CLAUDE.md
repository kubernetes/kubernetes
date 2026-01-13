# Package: projected

## Purpose
Implements the Projected volume plugin that combines multiple volume sources (Secret, ConfigMap, DownwardAPI, ServiceAccountToken) into a single volume mount.

## Key Types/Structs
- `projectedPlugin` - VolumePlugin for projected volumes
- `projectedVolumeMounter` - Handles mounting combined projected data
- `projectedVolumeUnmounter` - Handles cleanup

## Key Functions
- `ProbeVolumePlugins()` - Returns the projected plugin
- `SetUpAt()` - Collects and writes all projected sources
- `CollectData()` - Gathers data from all configured sources into file map
- `MakePayload()` - Assembles projected data with file modes

## Design Patterns
- Single volume combining multiple sources (reduces volume count)
- Supports Secret, ConfigMap, DownwardAPI, and ServiceAccountToken sources
- Wraps EmptyDir for underlying storage
- Uses AtomicWriter for atomic updates
- RequiresRemount returns true for source updates
- Common pattern for injecting service account tokens with bounded lifetime
