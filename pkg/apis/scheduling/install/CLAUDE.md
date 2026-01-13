# Package: install

## Purpose
Registers the scheduling API group with the Kubernetes API machinery, making PriorityClass and Workload resources available.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the scheduling API group with all versions

## Registration Order
1. Internal scheduling types
2. scheduling.k8s.io/v1 types
3. scheduling.k8s.io/v1beta1 types
4. scheduling.k8s.io/v1alpha1 types
5. Sets version priority: v1 > v1beta1 > v1alpha1

## Notes
- Auto-registers with legacy scheme via `init()`
- Uses `utilruntime.Must` to panic on registration failures
- All three API versions registered for backward compatibility
