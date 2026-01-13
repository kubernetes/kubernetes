# Package: storage

## Purpose
Provides REST storage implementation for Namespace objects including main resource, status, and finalize subresources with custom deletion logic.

## Key Types

- **REST**: Main storage for Namespace operations with custom Delete logic.
- **StatusREST**: Storage for /status subresource updates.
- **FinalizeREST**: Storage for /finalize subresource updates.

## Key Functions

- **NewREST(optsGetter)**: Returns REST, StatusREST, and FinalizeREST instances. Configures:
  - ReturnDeletedObject: true
  - ShouldDeleteDuringUpdate for finalizer-aware deletion
  - Separate stores with different strategies for status/finalize

- **Delete()**: Custom deletion logic that:
  - Validates UID/ResourceVersion preconditions
  - Sets DeletionTimestamp and phase to Terminating on first delete
  - Manages orphan/deleteDependent finalizers based on PropagationPolicy
  - Only performs actual deletion when finalizers are empty

- **ShouldDeleteNamespaceDuringUpdate()**: Checks if namespace should be deleted during update (requires empty finalizers).

- **ShortNames()**: Returns `["ns"]` for kubectl.

## Design Notes

- Implements complex namespace termination lifecycle with finalizers.
- Three-phase deletion: set terminating -> wait for finalizers -> actual delete.
- Shares underlying store between REST types for consistency.
- Implements `rest.StorageVersionProvider` and `rest.ResetFieldsStrategy`.
