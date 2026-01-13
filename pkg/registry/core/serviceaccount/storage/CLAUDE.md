# Package: storage

## Purpose
Provides REST storage implementation for ServiceAccount objects including the token request subresource.

## Key Types

- **REST**: Main storage for ServiceAccount operations with optional TokenREST.
- **TokenREST**: Implements /token subresource for generating bound service account tokens.

## Key Functions

- **NewREST(optsGetter, issuer, auds, max, podStorage, secretStorage, nodeStorage, extendExpiration, maxExtendedExpiration)**: Creates REST with optional TokenREST.
- **ShortNames()**: Returns `["sa"]`.
- **TokenREST.Create()**: Generates JWT token for ServiceAccount:
  - Validates request and looks up ServiceAccount
  - Supports binding to Pod, Node, or Secret
  - Enforces max expiration time
  - Extends expiration for projected tokens (safe transition)
  - Embeds node info for pod-bound tokens (ServiceAccountTokenPodNodeInfo feature)

## Design Notes

- Token subresource returns authentication.k8s.io/v1 TokenRequest.
- Supports bound tokens (to Pod, Node, Secret) with UID validation.
- Feature-gated: ServiceAccountTokenPodNodeInfo, ServiceAccountTokenNodeBinding, ServiceAccountTokenJTI.
- Token expiration can be automatically extended for backward compatibility.
- Validates that pod's ServiceAccountName matches the requested ServiceAccount.
