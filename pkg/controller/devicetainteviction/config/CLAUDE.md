# Package: config

Device taint eviction controller configuration types for the kube-controller-manager.

## Key Types

- `DeviceTaintEvictionControllerConfiguration`: Contains configuration elements including `ConcurrentSyncs` which controls the number of concurrent operations (pod deletion, ResourceClaim status updates).

## Purpose

Defines the internal configuration structure used by the device taint eviction controller. The default ConcurrentSyncs is 10.

## Design Notes

- Part of the component-config pattern used throughout kube-controller-manager
- Higher concurrency means faster processing but more API server and network load
