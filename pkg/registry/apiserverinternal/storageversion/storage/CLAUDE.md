# Package: storage

## Purpose
Provides REST storage implementation for StorageVersion resources in the Kubernetes API server internal API group. StorageVersion tracks the storage version of a resource stored in etcd.

## Key Types

- **REST**: Main REST storage struct embedding `genericregistry.Store` for StorageVersion CRUD operations
- **StatusREST**: REST endpoint for updating StorageVersion status subresource

## Key Functions

- **NewREST(optsGetter)**: Creates and returns REST and StatusREST instances configured with create/update/delete strategies from the storageversion strategy package
- **StatusREST.Get()**: Retrieves a StorageVersion object (required for Patch support)
- **StatusREST.Update()**: Updates only the status subset of a StorageVersion object
- **StatusREST.GetResetFields()**: Returns fields that should be reset during status updates

## Design Notes

- Uses the generic registry pattern common across Kubernetes API resources
- Status updates are handled separately to prevent spec modifications during status-only updates
- The StatusREST shares the underlying store with REST but uses a different update strategy
- Implements table conversion for kubectl output formatting
