# Package: podcertificaterequest

## Purpose
Implements the registry strategy for PodCertificateRequest resources, which allow pods to request X.509 certificates tied to their identity.

## Key Types

- **Strategy**: Base strategy for PodCertificateRequest objects
- **StatusStrategy**: Strategy for /status subresource with authorization checks

## Key Functions

- **NewStrategy()**: Creates a new Strategy instance
- **NewStatusStrategy(strategy, authorizer, clock)**: Creates StatusStrategy with authorizer
- **NamespaceScoped()**: Returns true - namespace-scoped (unlike CSR)
- **PrepareForCreate()**: Clears status
- **PrepareForUpdate()**: Preserves status from old object
- **StatusStrategy.PrepareForUpdate()**: Preserves spec, resets object meta for status
- **StatusStrategy.ValidateUpdate()**: Validates status update AND checks "sign" authorization

## Design Notes

- Namespace-scoped resource (tied to pod identity)
- Status updates require "sign" permission on the signer name
- Uses certauthorization.IsAuthorizedForSignerName for authorization checks
- Feature gated by PodCertificateRequest feature gate
- AllowUnconditionalUpdate returns false (requires resource version)
- Newer resource (2024) compared to CSR
