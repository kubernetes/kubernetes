# Package: devicetainteviction

Controller for evicting pods based on DRA (Dynamic Resource Allocation) device taints.

## Key Types

- `Controller`: Listens to taint changes on DRA devices and toleration changes on ResourceClaims, then deletes Pods that use ResourceClaims not tolerating NoExecute taints

## Key Functions

- `NewController()`: Creates the device taint eviction controller
- `Run()`: Starts the controller's reconciliation loop

## Purpose

Implements taint-based eviction for pods using DRA resources. When a device gets a NoExecute taint that the associated ResourceClaim doesn't tolerate, this controller evicts the pod. This is analogous to node taint eviction but for DRA devices.

## Key Features

- Watches ResourceSlices for device taints
- Watches ResourceClaims for toleration changes
- Processes DeviceTaintRules to apply taints
- Uses TimedWorkerQueue for scheduling evictions
- Updates DeviceTaintRule status during eviction

## Design Notes

- Event handlers run synchronously without blocking calls
- All blocking operations (pod deletion, status updates) happen in the TimedWorkerQueue
- ResourceSliceTracker handles applying DeviceTaintRule taints to ResourceSlices
