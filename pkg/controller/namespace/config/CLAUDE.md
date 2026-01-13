# Package: config

Namespace controller configuration types for the kube-controller-manager.

## Key Types

- `NamespaceControllerConfiguration`: Contains configuration elements including:
  - `NamespaceSyncPeriod`: Period for syncing namespace lifecycle updates
  - `ConcurrentNamespaceSyncs`: Number of namespace objects that can sync concurrently

## Purpose

Defines the internal configuration structure used by the Namespace controller.

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- Sync period affects how often namespaces are reprocessed even without changes
