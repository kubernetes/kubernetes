# Package: storage

## Purpose
Defines internal (unversioned) API types for the storage.k8s.io API group, covering storage classes, volume attachments, CSI drivers/nodes, and storage capacity.

## Key Types/Structs
- `StorageClass`: Defines a named class of storage with provisioner, parameters, reclaim policy, mount options, and volume binding mode
- `VolumeAttachment`: Captures intent to attach/detach volumes to nodes, with spec and status
- `CSIDriver`: Describes a Container Storage Interface volume driver deployed on the cluster
- `CSINode`: Holds information about CSI drivers installed on a node
- `CSIStorageCapacity`: Stores capacity information for a StorageClass in a topology segment
- `VolumeAttributesClass`: Specification of mutable volume attributes defined by CSI driver

## Key Constants
- `VolumeBindingImmediate`: PVCs should be immediately provisioned and bound
- `VolumeBindingWaitForFirstConsumer`: PVCs wait for first consumer pod before binding
- `VolumeLifecyclePersistent` / `VolumeLifecycleEphemeral`: CSI volume lifecycle modes
- `FSGroupPolicy` variants: `ReadWriteOnceWithFSType`, `File`, `None`

## Design Notes
- This is the internal representation; external versions are in v1, v1alpha1, v1beta1
- CSIDriver spec includes attach requirements, pod info injection, token requests, SELinux mount support
- VolumeAttachmentSource can reference either a PersistentVolume name or inline volume spec
- CSINodeDriver tracks per-node driver information including nodeID and topology keys
