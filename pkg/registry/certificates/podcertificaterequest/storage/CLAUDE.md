# Package: storage

## Purpose
Provides REST storage implementation for PodCertificateRequest resources with status subresource.

## Key Types

- **REST**: Main REST storage for PodCertificateRequest CRUD operations
- **StatusREST**: REST endpoint for /status subresource

## Key Functions

- **NewREST(optsGetter, authorizer, clock)**: Creates REST and StatusREST instances
  - Requires authorizer for status update authorization checks
  - Requires clock for timestamp validation
- **getAttrs()**: Returns labels and selectable fields (spec.signerName, spec.podName, spec.nodeName)
- **StatusREST.Get/Update()**: Standard status subresource operations

## Design Notes

- Namespace-scoped storage (unlike CSR which is cluster-scoped)
- Supports field selectors on spec.signerName, spec.podName, spec.nodeName
- Status strategy requires authorizer (checks "sign" permission on signerName)
- StatusREST implements Patcher interface
- Feature gated by PodCertificateRequest feature gate
