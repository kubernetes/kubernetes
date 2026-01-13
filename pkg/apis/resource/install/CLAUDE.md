# Package: install

## Purpose
Registers the resource API group with the Kubernetes API machinery, making DRA resources (ResourceClaim, DeviceClass, ResourceSlice, etc.) available.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the resource API group with all versions

## Registration Order
1. Internal resource types
2. resource.k8s.io/v1alpha3 types
3. resource.k8s.io/v1beta1 types
4. resource.k8s.io/v1beta2 types
5. resource.k8s.io/v1 types
6. Sets version priority: v1 > v1beta2 > v1beta1 > v1alpha3

## Notes
- Auto-registers with legacy scheme via `init()`
- Uses `utilruntime.Must` to panic on registration failures
- Four API versions reflect rapid DRA development
