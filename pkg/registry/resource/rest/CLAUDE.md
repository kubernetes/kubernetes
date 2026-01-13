# Package: rest

## Purpose
Provides the RESTStorageProvider for the resource.k8s.io API group, which wires up all DRA-related resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements genericapiserver.RESTStorageProvider for the resource API group.

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter, nsClient)**: Creates APIGroupInfo with storage for all resource API versions.
- **v1Storage(...)**: Creates storage map for v1 resources (ResourceClaim, ResourceClaimTemplate, DeviceClass, ResourceSlice).
- **v1alpha3Storage(...)**: Creates storage map for v1alpha3 resources (DeviceTaintRule).
- **GroupName()**: Returns "resource.k8s.io".

## Registered Resources

- **v1**: resourceclaims, resourceclaims/status, resourceclaimtemplates, deviceclasses, resourceslices
- **v1alpha3**: devicetaintrules, devicetaintrules/status

## Design Notes

- Checks apiResourceConfigSource to conditionally register resources.
- Requires namespace client for ResourceClaim and ResourceClaimTemplate validation.
