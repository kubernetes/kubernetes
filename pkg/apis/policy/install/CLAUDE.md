# Package: install

## Purpose
Registers the policy API group with the Kubernetes API machinery, making PodDisruptionBudget and Eviction resources available.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the policy API group with all versions

## Registration Order
1. Internal policy types
2. policy/v1beta1 types
3. policy/v1 types
4. Sets version priority: v1 > v1beta1

## Notes
- Auto-registers with legacy scheme via `init()`
- Uses `utilruntime.Must` to panic on registration failures
- Follows standard Kubernetes API installation pattern
