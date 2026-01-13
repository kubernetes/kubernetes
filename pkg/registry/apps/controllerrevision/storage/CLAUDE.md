# Package: storage

## Purpose
Provides REST storage implementation for ControllerRevision resources, enabling CRUD operations against etcd storage.

## Key Types

- **REST**: Main REST storage struct embedding `genericregistry.Store` for ControllerRevision operations

## Key Functions

- **NewREST(optsGetter)**: Creates and returns a REST storage instance configured with:
  - NewFunc/NewListFunc for creating ControllerRevision objects
  - Create/Update/Delete strategies from the controllerrevision strategy package
  - Table converter for kubectl output formatting

## Design Notes

- Simple storage implementation without subresources (no status or scale endpoints)
- Uses the generic registry Store pattern
- ControllerRevisions are primarily read/written by controllers, not end users
- Part of the apps/v1 API group
