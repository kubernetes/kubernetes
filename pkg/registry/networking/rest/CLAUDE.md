# Package: rest

Provides the REST storage provider for the `networking.k8s.io` API group.

## Key Types

- **RESTStorageProvider**: Implements `genericapiserver.RESTStorageProvider` for networking resources.

## Key Functions

- **NewRESTStorage**: Creates APIGroupInfo with all networking resource storage.
- **v1Storage**: Configures v1 API storage for:
  - NetworkPolicies
  - Ingresses (with status subresource)
  - IngressClasses
  - IPAddresses
  - ServiceCIDRs (with status subresource)
- **v1beta1Storage**: Configures v1beta1 API storage for IPAddresses and ServiceCIDRs.
- **GroupName**: Returns "networking.k8s.io".

## Design Notes

- Acts as the central wiring point for all networking API resources.
- Supports both v1 and v1beta1 API versions with appropriate resource mappings.
- Resources are conditionally registered based on APIResourceConfigSource settings.
- Ingress and ServiceCIDR have status subresources; others are spec-only.
