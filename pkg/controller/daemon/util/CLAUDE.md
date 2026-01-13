# Package: util

Utility functions for the DaemonSet controller.

## Key Functions

- `GetTemplateGeneration()`: Extracts template generation from the deprecated annotation on a DaemonSet
- `AddOrUpdateDaemonPodTolerations()`: Applies necessary tolerations to DaemonSet pods so they survive node conditions like NotReady, Unreachable, DiskPressure, MemoryPressure, PIDPressure, and Unschedulable
- `IsPodUpdated()`: Checks if a pod matches the current DaemonSet template
- `SurgeCount()`: Calculates the number of surge pods allowed during rolling update
- `UnavailableCount()`: Calculates the number of unavailable pods allowed during rolling update

## Purpose

Provides helper functions used by the DaemonSet controller for managing DaemonSet pods. Key functionality includes handling tolerations that allow DaemonSet pods to run on nodes even when those nodes have problems, and supporting rolling update strategies.

## Design Notes

- DaemonSet pods are designed to tolerate node conditions that would normally evict regular pods
- The toleration logic ensures system-critical DaemonSet pods stay running during node issues
- Supports both RollingUpdate and OnDelete update strategies
