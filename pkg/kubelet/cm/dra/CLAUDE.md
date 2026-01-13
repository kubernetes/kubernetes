# Package dra

Package dra implements Dynamic Resource Allocation (DRA) support in the kubelet, managing ResourceClaims for pods that need dynamically allocated resources like GPUs or accelerators.

## Key Types

- `Manager`: Manages ResourceClaim preparation/unprepare lifecycle
- `ClaimInfo`: Cached information about a prepared claim
- `ContainerInfo`: CDI devices for a container

## Manager Methods

- `NewManager`: Creates a new DRA manager with claim cache and health tracking
- `Start`: Starts the reconcile loop
- `PrepareResources`: Prepares all ResourceClaims for a pod (calls NodePrepareResources)
- `UnprepareResources`: Unprepares claims when pod terminates (calls NodeUnprepareResources)
- `GetResources`: Returns CDI devices for a container
- `GetContainerClaimInfos`: Returns claim info for a container's claims
- `UpdateAllocatedResourcesStatus`: Updates health status in pod status
- `HandleWatchResourcesStream`: Processes health updates from DRA drivers
- `Updates`: Channel for health update notifications

## Constants

- `draManagerStateFileName`: "dra_manager_state"
- `defaultReconcilePeriod`: 60 seconds
- `defaultWipingDelay`: 30 seconds (grace period for driver restart)

## Reconciliation

- Periodically syncs claim state with active pods
- Unprepares claims for terminated pods
- Handles driver registration/deregistration

## Design Notes

- Claims prepared via NodePrepareResources gRPC to DRA drivers
- CDI (Container Device Interface) devices passed to container runtime
- Health monitoring via WatchResources streaming RPC
- Checkpoint-based persistence for crash recovery
