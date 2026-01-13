# Package: storageversiongc

Implements the StorageVersion garbage collection controller that cleans up stale storage version entries.

## Key Types

- **Controller**: Watches kube-apiserver leases and StorageVersions, deleting stale entries.

## Key Functions

- **NewStorageVersionGC**: Creates a new controller with lease and storage version informers.
- **Run**: Starts workers for lease and storage version processing.
- **processDeletedLease**: Removes storage version entries for deleted API server leases.
- **syncStorageVersion**: Cleans up storage versions with invalid (non-existent) API server IDs.
- **updateOrDeleteStorageVersion**: Updates or deletes a storage version based on remaining server entries.

## Key Concepts

- StorageVersion tracks which version each API server uses for storing a resource.
- When an API server is removed, its storage version entries become stale.
- Controller watches for identity lease deletions to trigger cleanup.
- Also handles storage versions created with non-existent API server IDs.

## Design Patterns

- Uses separate queues for lease and storage version processing.
- Identifies API servers via identity leases in kube-system namespace.
- Validates leases have the correct component label (kube-apiserver).
- Deletes entire StorageVersion if all server entries are stale.
