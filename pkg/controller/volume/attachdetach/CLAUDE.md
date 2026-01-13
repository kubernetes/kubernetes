# Package: attachdetach

## Purpose
Implements the Attach/Detach controller that manages volume attach and detach operations for the cluster. It ensures volumes are attached to nodes before pods can use them and detached when no longer needed.

## Key Types

- **AttachDetachController**: Interface defining Run() and GetDesiredStateOfWorld() operations.
- **attachDetachController**: Main controller implementation containing caches, informers, and sub-components.
- **TimerConfig**: Configuration for internal reconciliation timers.

## Key Functions

- **NewAttachDetachController(...)**: Creates a new controller with all required informers and plugins.
- **Run(ctx)**: Starts the controller, initializes state, and runs reconciliation loops.
- **populateActualStateOfWorld**: Initializes ASW from node status and VolumeAttachments.
- **populateDesiredStateOfWorld**: Initializes DSW from existing pods.
- **podAdd/podUpdate/podDelete**: Event handlers for pod changes.
- **nodeAdd/nodeUpdate/nodeDelete**: Event handlers for node changes.

## Key Components

- **desiredStateOfWorld**: Tracks what volumes should be attached to which nodes.
- **actualStateOfWorld**: Tracks what volumes are currently attached.
- **reconciler**: Compares DSW vs ASW and triggers attach/detach operations.
- **desiredStateOfWorldPopulator**: Keeps DSW in sync with pod informer.
- **nodeStatusUpdater**: Updates Node.Status.VolumesAttached.

## Design Notes

- Implements `volume.VolumeHost` interface for volume plugin initialization.
- Processes detaches before attaches to handle pod rescheduling efficiently.
- Uses custom pod indexer (PodPVCIndex) for efficient PVC-to-pod lookups.
- Supports CSI migration for in-tree volume plugins.
- Default timer config: 100ms reconciler loop, 6min max unmount wait, 1min DSW populator loop.
