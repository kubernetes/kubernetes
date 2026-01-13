# Package: populator

## Purpose
Implements the DesiredStateOfWorldPopulator that keeps the attach/detach controller's desired state of world cache in sync with the pod informer as the source of truth.

## Key Types

- **DesiredStateOfWorldPopulator**: Interface with Run(ctx) method.
- **desiredStateOfWorldPopulator**: Implementation that periodically syncs DSW with pod informer.

## Key Functions

- **NewDesiredStateOfWorldPopulator(...)**: Creates a new populator instance.
- **Run(ctx)**: Starts the populator loop.
- **populatorLoopFunc**: Main loop function that removes deleted pods and adds active pods.
- **findAndRemoveDeletedPods**: Removes pods from DSW that no longer exist in the informer or have been terminated.
- **findAndAddActivePods**: Lists all pods and adds their volumes to DSW.

## Design Notes

- Two configurable intervals:
  - `loopSleepDuration`: Frequency of checking for deleted pods.
  - `listPodsRetryDuration`: Minimum time between full pod list operations.
- Removed pods are detected by comparing DSW entries against pod informer.
- Also removes volumes that change from attachable to non-attachable (e.g., CSIDriver changes).
- Skips terminated pods when adding volumes.
- Supports CSI migration through csiMigratedPluginManager and intreeToCSITranslator.
