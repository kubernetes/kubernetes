# Package: ttl

## Purpose
The TTL controller sets TTL annotations on nodes based on cluster size. These annotations are consumed by Kubelets as suggestions for how long to cache objects (secrets, config maps) before refetching from the API server.

## Key Types

- **Controller**: The main controller struct that watches node changes and updates TTL annotations.
- **ttlBoundary**: Defines cluster size boundaries and corresponding TTL values.

## Key Functions

- **NewTTLController(ctx, nodeInformer, kubeClient)**: Creates a new TTL controller instance.
- **Run(ctx, workers)**: Starts the controller workers to process node updates.
- **addNode/updateNode/deleteNode**: Event handlers that track cluster size and trigger TTL updates.
- **updateNodeIfNeeded(ctx, key)**: Patches the node with the appropriate TTL annotation if it differs from desired.
- **patchNodeWithAnnotation**: Applies the TTL annotation via strategic merge patch.

## Design Notes

- TTL values scale with cluster size using predefined boundaries:
  - 0-100 nodes: 0s TTL
  - 90-500 nodes: 15s TTL
  - 450-1000 nodes: 30s TTL
  - 900-2000 nodes: 60s TTL
  - 1800+ nodes: 300s TTL
- Boundaries overlap to provide hysteresis and prevent thrashing when cluster size fluctuates.
- Uses a rate-limited work queue for processing node updates.
- This is considered a temporary workaround until Kubelets can watch specific secrets/configmaps.
