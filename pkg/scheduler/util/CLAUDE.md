# Package: util

## Purpose
Provides utility functions for the scheduler, including pod resource calculations and common helper operations.

## Key Types
- `PodResourcesOptions` - Options for calculating pod resource requirements

## Key Functions
- `GetPodResourceRequest()` - Calculates total resource requests for a pod
- `GetContainerResourceRequest()` - Gets resource request for a single container
- Various helper utilities for scheduler operations

## Design Patterns
- Pure utility functions with no state
- Handles both regular containers and init containers
- Accounts for sidecar containers and resource overhead
- Provides consistent resource calculation across scheduler components
