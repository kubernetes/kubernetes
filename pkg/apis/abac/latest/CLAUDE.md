# Package: latest

## Purpose
Initialization package that imports all ABAC API versions to register them with the runtime scheme.

## Imports
- `k8s.io/kubernetes/pkg/apis/abac` - Internal (hub) version
- `k8s.io/kubernetes/pkg/apis/abac/v0` - Legacy v0 external version
- `k8s.io/kubernetes/pkg/apis/abac/v1beta1` - v1beta1 external version

## Design Notes
- Uses blank imports (`_`) to trigger the `init()` functions in each package
- These init functions register types with the scheme
- This pattern ensures all versions are available when working with ABAC policies
- TODO comment notes this file structure needs refactoring to match other latest packages
