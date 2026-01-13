# Package: garbagecollector

Kubernetes garbage collector controller for cascading deletion.

## Key Types

- `GarbageCollector`: Main controller that deletes objects based on owner references
- `GraphBuilder`: Builds and maintains the dependency graph of objects

## Key Functions

- `NewGarbageCollector()`: Creates the garbage collector with dynamic client
- `Run()`: Starts the garbage collection workers
- `Sync()`: Synchronizes monitors for all API resources
- `attemptToDeleteItem()`: Processes deletion of a single object
- `orphanDependents()`: Removes owner references from dependents (orphan policy)

## Purpose

Implements cascading deletion for Kubernetes objects. When an object is deleted, the garbage collector deletes or orphans its dependents based on the deletion propagation policy (Foreground, Background, or Orphan).

## Key Features

- Dependency graph tracking via owner references
- Foreground deletion: delete dependents before owner
- Background deletion: delete owner immediately, dependents asynchronously
- Orphan deletion: remove owner references, keep dependents
- Virtual node support for handling out-of-order events

## Design Notes

- Uses a dependency graph (graph.go) to track object relationships
- GraphBuilder is single-threaded for graph writes
- Multiple workers process attemptToDelete and attemptToOrphan queues
- Monitors all API resources via dynamic informers
