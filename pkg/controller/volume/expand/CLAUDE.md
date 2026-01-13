# Package: expand

## Purpose
Implements the Expand controller that handles volume expansion for PersistentVolumeClaims. This controller is deprecated and primarily exists to add necessary annotations for external volume expansion.

## Key Types

- **ExpandController**: Interface with Run(ctx) method.
- **CSINameTranslator**: Interface for getting CSI driver names from in-tree plugin names.
- **expandController**: Implementation that watches PVCs and triggers expansion operations.

## Key Functions

- **NewExpandController(ctx, kubeClient, pvcInformer, plugins, translator, csiMigratedPluginManager)**: Creates a new expand controller.
- **Run(ctx)**: Starts the controller with 10 workers.
- **enqueuePVC(obj)**: Enqueues bound PVCs when their request size increases or capacity changes.
- **syncHandler(ctx, key)**: Main processing logic for PVC expansion.
- **expand(logger, pvc, pv, resizerName)**: Performs the actual expansion operation.
- **isNodeExpandComplete(logger, pvc, pv)**: Checks if node-side expansion is complete.

## Key Behaviors

1. **CSI Migration**: If volume is migratable to CSI, sets resizer annotation and delegates to external resizer.
2. **In-tree Expansion**: For non-migrated volumes, finds expandable plugin and runs expansion.
3. **Annotation Management**: Removes `AnnPreResizeCapacity` annotation when expansion completes.
4. **Recovery Support**: Supports `RecoverVolumeExpansionFailure` feature gate for expansion recovery.

## Design Notes

- Implements `volume.VolumeHost` interface (mostly stub methods).
- Default worker count: 10.
- Enqueues PVCs on: request size increase, capacity increase, or deletion.
- Only processes bound PVCs.
- Controller is deprecated; expansion is moving to external resizers.
