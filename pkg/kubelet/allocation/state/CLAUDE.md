# Package: state

## Purpose
This package provides state management interfaces and implementations for tracking pod resource allocations. It supports both checkpoint-based persistence and in-memory storage for testing.

## Key Types

- **PodResourceInfo**: Stores resource requirements for containers within a pod
- **PodResourceInfoMap**: Maps pod UIDs to their PodResourceInfo
- **Reader**: Interface for reading pod resource state
- **State**: Combined interface for reading and writing pod resource state

## Key Interfaces

- **GetContainerResources()**: Gets allocated resources for a specific container
- **GetPodResourceInfoMap()**: Gets all pod resource allocations
- **GetPodLevelResources()**: Gets pod-level resource allocation
- **SetPodResourceInfo()**: Sets resource info for a pod
- **RemovePod()**: Removes state for a pod
- **RemoveOrphanedPods()**: Cleans up state for deleted pods

## Implementations

- **stateCheckpoint**: Persistent state using kubelet checkpoint manager
- **stateMemory**: In-memory state for testing
- **noopStateCheckpoint**: No-op implementation when feature is disabled

## Design Notes

- PodResourceInfo tracks both container-level and pod-level resources
- Clone() method performs deep copy of all resource requirements
- Used by allocation manager to persist allocations across restarts
- Supports the InPlacePodVerticalScaling and InPlacePodLevelResourcesVerticalScaling features
