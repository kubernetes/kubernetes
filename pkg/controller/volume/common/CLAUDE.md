# Package: common

## Purpose
Provides shared utilities for volume controllers, primarily for creating pod indexers that enable efficient lookups of pods by their PVC references.

## Key Constants

- **PodPVCIndex**: Index name ("pod-pvc-index") for the pod-to-PVC indexer.

## Key Functions

- **PodPVCIndexFunc()**: Returns an index function that extracts PVC keys ("namespace/name") from a pod. Handles both:
  - Regular PVC references via `pod.Spec.Volumes[].PersistentVolumeClaim`
  - Ephemeral volumes via `pod.Spec.Volumes[].Ephemeral` (generates PVC name using ephemeral.VolumeClaimName)
- **AddPodPVCIndexerIfNotPresent(indexer)**: Adds the PodPVC index function to an indexer if not already present.
- **AddIndexerIfNotPresent(indexer, indexName, indexFunc)**: Generic helper to add an indexer only if it does not exist.

## Design Notes

- Used by attach/detach controller, ephemeral controller, and other volume-related controllers.
- Enables efficient reverse lookups: given a PVC, find all pods that reference it.
- Thread-safe: checks for existing indexer before adding.
