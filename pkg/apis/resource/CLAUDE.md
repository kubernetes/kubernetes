# Package: resource

## Purpose
Defines the internal (unversioned) API types for the resource.k8s.io API group, implementing Dynamic Resource Allocation (DRA) for hardware devices like GPUs, FPGAs, and accelerators.

## Key Types

### ResourceSlice
Represents resources in a pool managed by a DRA driver. Contains devices with attributes and capacities.
- `ResourceSliceSpec`: Driver name, pool, node selection (NodeName/NodeSelector/AllNodes), devices list
- `Device`: Name, attributes, capacities, taints, counter consumption
- `DeviceAttribute`: Int, Bool, String, or Version value
- `DeviceCapacity`: Quantity with optional request policy

### ResourceClaim
A request for device resources, similar to PersistentVolumeClaim for storage.
- `ResourceClaimSpec`: Contains `DeviceClaim` with requests, constraints, and config
- `ResourceClaimStatus`: Allocation result, reserved consumers, device status
- `DeviceRequest`: Named request for devices with selectors and allocation mode (ExactCount/All)

### DeviceClass
Cluster-scoped configuration defining device selectors and driver-specific config.
- `DeviceClassSpec`: Selectors, config, optional extended resource name
- `DeviceClassConfiguration`: Opaque driver-specific parameters

### ResourceClaimTemplate
Template for creating ResourceClaim objects for pods.

### DeviceTaintRule
Cluster-scoped rule to add taints to devices matching a selector.

### Allocation Types
- `AllocationResult`: Devices allocated, node selector
- `DeviceRequestAllocationResult`: Per-request allocation (driver, pool, device, tolerations)

## Key Constants
- `Finalizer`: "resource.kubernetes.io/delete-protection"
- `DeviceAllocationModeExactCount`, `DeviceAllocationModeAll`
- `DeviceTaintEffectNone`, `DeviceTaintEffectNoSchedule`, `DeviceTaintEffectNoExecute`

## CEL Device Selection
Uses CEL expressions to match devices based on attributes/capacities:
```cel
device.attributes["dra.example.com"].model == "A100"
```

## Notes
- Requires `DynamicResourceAllocation` feature gate
- Driver names follow CSI driver naming (DNS subdomain)
- Devices identified by tuple: `<driver>/<pool>/<device>`
