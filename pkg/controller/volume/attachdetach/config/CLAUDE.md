# Package: config

## Purpose
Defines the internal configuration types for the Attach/Detach controller used by the kube-controller-manager.

## Key Types

- **AttachDetachControllerConfiguration**: Configuration struct containing:
  - `DisableAttachDetachReconcilerSync`: Disables periodic reconciliation sync (default: false, meaning enabled).
  - `ReconcilerSyncLoopPeriod`: Time between reconciler sync executions (default: 60s).
  - `DisableForceDetachOnTimeout`: Disables force detach when unmount timeout expires (default: false, meaning force detach is enabled).

## Design Notes

- Uses `metav1.Duration` for time configuration fields.
- Part of the component config pattern used across Kubernetes controllers.
- Force detach handles crashed/unavailable nodes by detaching volumes after timeout.
