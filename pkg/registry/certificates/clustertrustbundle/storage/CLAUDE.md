# Package: storage

## Purpose
Provides REST storage implementation for ClusterTrustBundle resources.

## Key Types

- **REST**: Main REST storage embedding `genericregistry.Store` for ClusterTrustBundle CRUD operations

## Key Functions

- **NewREST(optsGetter)**: Creates REST instance with configured strategies
- **getAttrs()**: Returns labels and selectable fields (including spec.signerName) for filtering

## Design Notes

- Simple storage implementation without subresources
- No status subresource (ClusterTrustBundles are simple data containers)
- Supports field selector on spec.signerName for filtering bundles by signer
- Implements StandardStorage, TableConvertor, and GenericStore interfaces
- Feature gated by ClusterTrustBundle feature gate
