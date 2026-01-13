# Package: install

## Purpose
Registers the networking API group with the Kubernetes API machinery, making NetworkPolicy, Ingress, and related types available.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the networking API group with internal types and versioned APIs (v1, v1beta1)

## Registration Order
1. Internal networking types
2. networking.k8s.io/v1 types
3. networking.k8s.io/v1beta1 types
4. Sets version priority: v1 > v1beta1

## Notes
- Auto-registers with legacy scheme via `init()`
- Uses `utilruntime.Must` to panic on registration failures
- Follows standard Kubernetes API installation pattern
