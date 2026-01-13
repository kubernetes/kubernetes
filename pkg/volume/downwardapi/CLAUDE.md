# Package: downwardapi

## Purpose
Implements the DownwardAPI volume plugin that exposes pod and container metadata as files.

## Key Types/Structs
- `downwardAPIPlugin` - VolumePlugin for DownwardAPI volumes
- `downwardAPIVolumeMounter` - Handles mounting DownwardAPI data
- `downwardAPIVolumeUnmounter` - Handles cleanup

## Key Functions
- `ProbeVolumePlugins()` - Entry point for plugin detection
- `CollectData()` - Gathers pod field references and resource field refs into file projections
- `SetUpAt()` - Writes DownwardAPI data using AtomicWriter

## Design Patterns
- Wraps EmptyDir for underlying storage
- Supports pod field references (metadata.name, metadata.namespace, etc.)
- Supports container resource field references (requests/limits)
- RequiresRemount returns true for dynamic updates
- Uses AtomicWriter for safe updates
