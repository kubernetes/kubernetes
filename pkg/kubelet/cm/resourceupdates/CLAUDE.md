# Package resourceupdates

Package resourceupdates defines types for communicating resource status changes between kubelet components.

## Key Types

- `Update`: Represents a resource status change notification

## Update Fields

- `PodUIDs`: List of pod UIDs whose status needs updating

## Usage

Used by device manager and DRA manager to notify the kubelet when device health or resource status changes, triggering pod status updates.

## Design Notes

- Simple struct for decoupled component communication
- May be extended with container name, resource name, and new status fields
- Channel-based delivery pattern for async notifications
