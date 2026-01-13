# Package: deletion

Namespace resource deletion implementation.

## Key Types

- `NamespacedResourcesDeleterInterface`: Interface for deleting all resources in a namespace
- `namespacedResourcesDeleter`: Implementation that handles complete namespace cleanup
- `NamespaceConditionUpdater`: Translates deletion errors into namespace status conditions

## Key Functions

- `NewNamespacedResourcesDeleter()`: Creates the deleter with necessary clients
- `Delete()`: Main entry point that orchestrates namespace deletion

## Purpose

Implements the actual deletion of all resources within a namespace. This includes discovering all API resources, iterating through them, deleting all instances, and finally removing the namespace finalizer.

## Key Features

- Discovers all namespaced API resources dynamically
- Handles deletion errors gracefully with status conditions
- Tracks remaining resources and finalizers in namespace status
- Supports retry on conflict for concurrent updates

## Namespace Conditions

- `NamespaceDeletionDiscoveryFailure`: API discovery failed
- `NamespaceDeletionGVParsingFailure`: GroupVersion parsing failed
- `NamespaceDeletionContentFailure`: Content deletion failed
- `NamespaceContentRemaining`: Resources still exist
- `NamespaceFinalizersRemaining`: Finalizers blocking deletion

## Design Notes

- Uses metadata client for efficient bulk deletion
- Caches unsupported operations to avoid repeated failures
- Returns ResourcesRemainingError when deletion is in progress
