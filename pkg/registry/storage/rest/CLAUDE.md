# Package: rest

## Purpose
Provides the RESTStorageProvider for the storage.k8s.io API group, which wires up all storage-related resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements genericapiserver.RESTStorageProvider for the storage API group.

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage for all storage API versions.
- **v1Storage(...)**: Creates storage map for v1 resources (StorageClass, VolumeAttachment, CSINode, CSIDriver, CSIStorageCapacity, VolumeAttributesClass).
- **v1beta1Storage(...)**: Creates storage map for v1beta1 resources (VolumeAttributesClass).
- **GroupName()**: Returns "storage.k8s.io".

## Registered Resources

- **v1**: storageclasses, volumeattachments, volumeattachments/status, csinodes, csidrivers, csistoragecapacities, volumeattributesclasses
- **v1beta1**: volumeattributesclasses

## Design Notes

- Conditionally registers resources based on apiResourceConfigSource.
- VolumeAttachment is the only storage resource with a status subresource.
