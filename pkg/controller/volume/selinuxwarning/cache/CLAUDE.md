# Package: cache

## Purpose
Provides the VolumeCache interface and implementation for tracking volumes, their SELinux labels, and detecting conflicts between pods sharing volumes with different SELinux configurations.

## Key Types

- **VolumeCache**: Interface for managing volume-pod-SELinux associations.
- **volumeCache**: Thread-safe implementation storing volume-to-pod mappings.
- **usedVolume**: Represents a volume used by one or more pods.
- **podInfo**: Stores SELinux label and change policy for a pod.
- **Conflict**: Represents a conflict between two pods using the same volume with different properties.

## Key Functions

### VolumeCache Operations
- **NewVolumeLabelCache(seLinuxTranslator)**: Creates a new empty cache.
- **AddVolume(logger, volumeName, podKey, seLinuxLabel, changePolicy, csiDriver)**: Adds a volume and returns any conflicts detected.
- **DeletePod(logger, podKey)**: Removes a pod and prunes empty volume entries.
- **GetPodsForCSIDriver(driverName)**: Returns all pods using volumes from a specific CSI driver.
- **SendConflicts(logger, ch)**: Sends all current conflicts to a channel (for metrics scraping).

### Conflict Type
- **EventMessage()**: Generates a human-readable conflict description for events.

## Conflict Detection

Two types of conflicts are detected:
1. **SELinuxChangePolicyConflict**: Pods with different SELinuxChangePolicy values.
2. **SELinuxLabelConflict**: Pods with conflicting SELinux labels (using translator.Conflicts()).

## Design Notes

- Thread-safe via sync.RWMutex.
- Prunes empty entries automatically when pods are deleted.
- Namespace-aware conflict messages (hides cross-namespace pod names).
- Uses seLinuxTranslator to determine label conflicts.
