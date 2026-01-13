# Package: pvcprotection

## Purpose
Implements the PVC Protection controller that prevents deletion of PersistentVolumeClaims while they are actively used by pods. It manages the `kubernetes.io/pvc-protection` finalizer.

## Key Types

- **Controller**: Main controller watching PVCs and Pods to manage protection finalizers.
- **LazyLivePodList**: Lazy-loading cache for live pod list API calls (optimization).
- **pvcProcessingStore**: Batches PVCs by namespace for efficient processing.

## Key Functions

### Controller Lifecycle
- **NewPVCProtectionController(logger, pvcInformer, podInformer, client)**: Creates the controller with PodPVCIndex.
- **Run(ctx, workers)**: Starts main worker and namespace processing workers.

### Processing Logic
- **processPVC(ctx, namespace, pvcName, lazyLivePodList)**: Main processing for a single PVC.
- **isBeingUsed(ctx, pvc, lazyLivePodList)**: Checks if PVC is used by any pod.
- **askInformer(logger, pvc)**: Fast check via informer cache.
- **askAPIServer(ctx, pvc, lazyLivePodList)**: Live API check if cache check is inconclusive.
- **podUsesPVC(logger, pod, pvc)**: Checks if a specific pod references the PVC.

### Finalizer Management
- **addFinalizer(ctx, pvc)**: Adds protection finalizer to PVC.
- **removeFinalizer(ctx, pvc)**: Removes protection finalizer (allows deletion).

## Design Notes

- Two-tier checking: fast informer check, then live API check if needed.
- Uses PodPVCIndex for efficient PVC-to-pod lookups.
- Handles ephemeral volumes (ownership verification).
- Considers pods "not using" PVC if pod has deletion timestamp + grace period = 0.
- Batches PVCs by namespace to share live pod list cache.
- Works with admission plugin (adds finalizers to new PVCs).
