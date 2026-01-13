# Package: populator

## Purpose
The `populator` package implements the Desired State of World (DSW) populator that monitors pods and updates the DSW cache with volume requirements.

## Key Interfaces

- **DesiredStateOfWorldPopulator**: Interface for the populator.
  - `Run()`: Starts the populator loop.
  - `ReprocessPod()`: Forces reprocessing of a specific pod's volumes.
  - `HasAddedPods()`: Returns true if populator has processed pods at least once.

## Key Functions

- **NewDesiredStateOfWorldPopulator**: Creates a new populator instance.

## Key Types

- **desiredStateOfWorldPopulator**: Implementation that periodically syncs pods with DSW.

## Behavior

1. Periodically iterates through all pods from pod manager.
2. For each pod, determines required volumes from pod spec.
3. Adds volumes to DSW with appropriate mount options.
4. Removes volumes from DSW when pods no longer need them.
5. Handles SELinux context requirements for volumes.
6. Processes ephemeral volumes and persistent volume claims.

## Design Notes

- Runs as a background goroutine with configurable loop period.
- Coordinates with pod manager to get current pod list.
- Uses PVC lister to resolve persistent volume claims.
- Handles volume plugin lookups to determine mount behavior.
- Supports immediate reprocessing for specific pods via ReprocessPod().
