# Package: pvprotection

## Purpose
Implements the PV Protection controller that prevents deletion of PersistentVolumes while they are bound to PersistentVolumeClaims. It manages the `kubernetes.io/pv-protection` finalizer.

## Key Types

- **Controller**: Main controller watching PVs to manage protection finalizers.

## Key Functions

### Controller Lifecycle
- **NewPVProtectionController(logger, pvInformer, client)**: Creates the controller.
- **Run(ctx, workers)**: Starts the controller workers.

### Processing Logic
- **processPV(ctx, pvName)**: Main processing for a single PV.
- **isBeingUsed(pv)**: Checks if PV is bound (Status.Phase == VolumeBound).
- **pvAddedUpdated(logger, obj)**: Event handler that enqueues PVs needing finalizer changes.

### Finalizer Management
- **addFinalizer(ctx, pv)**: Adds protection finalizer to PV.
- **removeFinalizer(ctx, pv)**: Removes protection finalizer (allows deletion).

## Design Notes

- Simpler than PVC protection: only checks PV.Status.Phase (no pod lookups needed).
- PV is "in use" if Status.Phase is VolumeBound.
- Finalizer prevents accidental PV deletion while data is still needed.
- Admission plugin should add finalizers to new PVs; this controller handles legacy PVs.
- Deep copies PV before modification to avoid cache mutation.
