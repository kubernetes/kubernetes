# Package: config

Deployment controller configuration types for the kube-controller-manager.

## Key Types

- `DeploymentControllerConfiguration`: Contains configuration elements for the DeploymentController, including `ConcurrentDeploymentSyncs` which controls the number of deployment objects that can sync concurrently.

## Purpose

Defines the internal configuration structure used by the Deployment controller. This package provides the Go types that represent controller configuration, which are then converted to/from versioned API types (v1alpha1).

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- Higher concurrency means more responsive deployments but increased CPU and network load
