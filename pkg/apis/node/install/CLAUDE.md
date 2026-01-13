# Package: install

## Purpose
Registers the node API group with the Kubernetes API machinery, making RuntimeClass resources available.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the node API group with all versions

## Registration Order
1. Internal node types
2. node.k8s.io/v1alpha1 types
3. node.k8s.io/v1beta1 types
4. node.k8s.io/v1 types
5. Sets version priority: v1 > v1beta1

## Notes
- Auto-registers with legacy scheme via `init()`
- Uses `utilruntime.Must` to panic on registration failures
- v1alpha1 is registered but not in the priority list (deprecated)
