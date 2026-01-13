# Package: secret

## Purpose
Implements the Secret volume plugin that projects Kubernetes Secret data as files into pods.

## Key Types/Structs
- `secretPlugin` - VolumePlugin implementation for Secret volumes
- `secretVolumeMounter` - Handles mounting Secret data as files
- `secretVolumeUnmounter` - Handles cleanup of Secret volume mounts

## Key Functions
- `ProbeVolumePlugins()` - Entry point returning the secret plugin
- `MakePayload()` - Converts Secret data to file projections with modes
- `SetUpAt()` - Writes Secret data to the volume directory using AtomicWriter

## Design Patterns
- Wraps EmptyDir for underlying storage (uses tmpfs by default)
- Uses AtomicWriter for atomic updates of Secret data
- Supports optional Secrets (volume mounts succeed even if Secret missing)
- Supports custom file modes via Items[].Mode
- RequiresRemount returns true to support Secret updates
- Similar implementation pattern to ConfigMap volumes
