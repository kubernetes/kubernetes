# Package: vacprotection

## Purpose
Implements the VolumeAttributesClass (VAC) Protection controller that prevents deletion of VolumeAttributesClasses while they are referenced by PVs or PVCs. It manages the `kubernetes.io/vac-protection` finalizer.

## Key Types

- **Controller**: Main controller watching VACs, PVs, and PVCs to manage protection finalizers.

## Key Functions

### Controller Lifecycle
- **NewVACProtectionController(logger, client, pvcInformer, pvInformer, vacInformer)**: Creates the controller with custom indexers.
- **Run(ctx, workers)**: Starts the controller workers.

### Processing Logic
- **processVAC(ctx, vacName)**: Main processing for a single VAC.
- **isBeingUsed(ctx, vac)**: Checks if any PV or PVC references the VAC.
- **getPVsAssignedToVAC(vacName)**: Uses index to find PVs referencing the VAC.
- **getPVCsAssignedToVAC(vacName)**: Uses index to find PVCs referencing the VAC.

### Finalizer Management
- **addFinalizer(ctx, vac)**: Adds protection finalizer to VAC.
- **removeFinalizer(ctx, vac)**: Removes protection finalizer (allows deletion).

### Event Handlers
- **vacAddedUpdated**: Enqueues VACs needing finalizer changes.
- **pvcUpdated/pvcDeleted**: Re-evaluates VACs when PVC references change.
- **pvUpdated/pvDeleted**: Re-evaluates VACs when PV references change.

## Design Notes

- Uses informer-based checking only (no live API calls).
- Known race condition: may remove finalizer too early if PVC creation event hasn't reached informer yet.
- Custom indexer tracks VAC references in PVs/PVCs for efficient lookups.
- PVC can reference VAC via: Spec.VolumeAttributesClassName, Status.CurrentVolumeAttributesClassName, or Status.ModifyVolumeStatus.TargetVolumeAttributesClassName.
- PV references VAC via Spec.VolumeAttributesClassName.
