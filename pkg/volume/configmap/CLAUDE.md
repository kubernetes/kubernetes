# Package: configmap

## Purpose
Implements the ConfigMap volume plugin that projects ConfigMap data as files into pods.

## Key Types/Structs
- `configMapPlugin` - VolumePlugin implementation for ConfigMap volumes
- `configMapVolumeMounter` - Handles mounting ConfigMap data as files
- `configMapVolumeUnmounter` - Handles cleanup of ConfigMap volume mounts

## Key Functions
- `ProbeVolumePlugins()` - Entry point returning the configmap plugin
- `MakePayload()` - Converts ConfigMap data to file projections with modes
- `SetUpAt()` - Writes ConfigMap data to the volume directory using AtomicWriter

## Design Patterns
- Wraps EmptyDir for underlying storage (uses tmpfs)
- Uses AtomicWriter for atomic updates of ConfigMap data
- Supports optional ConfigMaps (volume mounts succeed even if ConfigMap missing)
- Supports custom file modes via Items[].Mode
- RequiresRemount returns true to support ConfigMap updates
