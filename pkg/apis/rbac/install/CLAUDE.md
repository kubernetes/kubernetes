# Package: install

## Purpose
Registers the rbac API group with the Kubernetes API machinery, making Role, ClusterRole, RoleBinding, and ClusterRoleBinding resources available.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the rbac API group with all versions

## Registration Order
1. Internal rbac types
2. rbac.authorization.k8s.io/v1 types
3. rbac.authorization.k8s.io/v1beta1 types
4. rbac.authorization.k8s.io/v1alpha1 types
5. Sets version priority: v1 > v1beta1 > v1alpha1

## Notes
- Auto-registers with legacy scheme via `init()`
- Uses `utilruntime.Must` to panic on registration failures
- All three external versions are registered and prioritized
