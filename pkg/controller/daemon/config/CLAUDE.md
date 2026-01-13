# Package: config

DaemonSet controller configuration types for the kube-controller-manager.

## Key Types

- `DaemonSetControllerConfiguration`: Contains configuration elements for the DaemonSetController, including `ConcurrentDaemonSetSyncs` which controls the number of daemonset objects that can sync concurrently.

## Purpose

Defines the internal configuration structure used by the DaemonSet controller. This package provides the Go types that represent controller configuration, which are then converted to/from versioned API types (v1alpha1).

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- Configuration is typically loaded from a file or command-line flags and converted to these internal types
