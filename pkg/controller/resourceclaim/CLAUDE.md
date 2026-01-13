# Package: resourceclaim

Implements the ResourceClaim controller for Dynamic Resource Allocation (DRA).

## Key Types

- **Controller**: Watches pods with resource claims and creates/manages claims as needed.
- **podSchedulingState**: Tracks scheduling status for pods waiting for resources.

## Key Functions

- **NewController**: Creates a new resource claim controller.
- **Run**: Starts workers processing resource claim reconciliation.
- **syncPod**: Main sync logic - handles claim creation from templates and reservation management.
- **handleClaim**: Processes individual claims for a pod (create, delete, reserve).
- **handleResourceClaimTemplatePodResourceClaims**: Creates claims from ResourceClaimTemplates.

## Key Features

- Creates ResourceClaims from pod ResourceClaimTemplates.
- Manages claim reservations for pods.
- Cleans up orphaned claims when pods are deleted.
- Handles DeviceClass-specific scheduling requirements.
- Supports structured parameters for resource allocation.

## Design Patterns

- Uses pod indexer for efficient claim-to-pod lookups.
- Creates claims with owner references for automatic garbage collection.
- Handles both immediate and wait-for-first-consumer allocation modes.
- Exposes Prometheus metrics for claim operations.
