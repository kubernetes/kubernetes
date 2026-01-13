# Package: reconciler

## Purpose
The `reconciler` package implements the volume manager reconciler that drives the Actual State of World (ASW) toward the Desired State of World (DSW) by performing attach, mount, unmount, and detach operations.

## Key Interfaces

- **Reconciler**: Interface for the reconciliation loop.
  - `Run()`: Starts the reconciler loop.
  - `StatesHasBeenSynced()`: Returns true after initial reconciliation completes.

## Key Functions

- **NewReconciler**: Creates a new reconciler instance.

## Reconciliation Logic

1. **Unmount volumes**: For mounted volumes not in DSW, trigger unmount.
2. **Mount volumes**: For volumes in DSW not yet mounted, trigger mount.
3. **Attach volumes**: For volumes requiring attachment, wait for or trigger attach.
4. **Detach volumes**: For attached volumes no longer needed, trigger detach.
5. **Expand volumes**: For volumes needing expansion, trigger filesystem resize.

## Key Behaviors

- Handles uncertain attachment states after kubelet restart.
- Reconstructs volume state from on-disk mount points.
- Coordinates with attach/detach controller via node status.
- Supports both immediate and waiting reconciliation modes.
- Handles SELinux mount context changes.
- Implements exponential backoff for failed operations.

## Design Notes

- Runs as a background goroutine with configurable sync period.
- Uses operationexecutor for actual volume operations.
- Tracks in-progress operations to avoid duplicate work.
- Supports kubelet-managed and controller-managed attachment modes.
- Handles volume reconstruction for crash recovery.
