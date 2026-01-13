# Package: storage/validation

## Purpose
Provides validation logic for storage API types including StorageClass, VolumeAttachment, CSIDriver, CSINode, CSIStorageCapacity, and VolumeAttributesClass.

## Key Functions
- `ValidateStorageClass`: Validates StorageClass fields including provisioner, parameters, reclaim policy, and allowed topologies
- `ValidateVolumeAttachment`: Validates VolumeAttachment spec including attacher and source
- `ValidateCSIDriver`: Validates CSIDriver spec including attach requirements, volume lifecycle modes, and token requests
- `ValidateCSINode`: Validates CSINode including driver list and topology keys
- `ValidateCSIStorageCapacity`: Validates storage capacity objects
- `ValidateVolumeAttributesClass`: Validates mutable volume attributes

## Key Validation Rules
- Provisioner names must be valid qualified names
- VolumeBindingMode must be Immediate or WaitForFirstConsumer
- CSIDriver names must match the CSI GetPluginName() response format
- Topology keys must be valid label keys

## Design Notes
- Uses field.ErrorList pattern for accumulating validation errors
- Provides both create and update validation variants
- Validates cross-field dependencies and mutual exclusivity constraints
