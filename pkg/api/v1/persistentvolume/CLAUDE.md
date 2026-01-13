# Package: persistentvolume

## Purpose
Provides utilities for discovering secret references in v1.PersistentVolume objects, used for dependency tracking and garbage collection.

## Key Types
- `Visitor` - Function type `func(namespace, name string, kubeletVisible bool) (shouldContinue bool)` for visiting secret references

## Key Functions
- `VisitPVSecretNames(pv *corev1.PersistentVolume, visitor Visitor) bool` - Visits all secrets referenced by the PV spec, returns false if visiting was short-circuited

## Volume Types with Secret References
The function handles secrets from:
- AzureFile (SecretName with optional SecretNamespace)
- CephFS (SecretRef)
- Cinder (SecretRef)
- FlexVolume (SecretRef)
- RBD (SecretRef)
- ScaleIO (SecretRef)
- ISCSI (SecretRef)
- StorageOS (SecretRef)
- CSI (ControllerPublishSecretRef, ControllerExpandSecretRef, NodePublishSecretRef, NodeStageSecretRef, NodeExpandSecretRef)

## Design Notes
- The `kubeletVisible` parameter indicates whether the secret is visible to kubelets (true for node-level secrets, false for controller-level CSI secrets)
- For volumes without explicit namespace, falls back to ClaimRef namespace
- Empty secret names are skipped
