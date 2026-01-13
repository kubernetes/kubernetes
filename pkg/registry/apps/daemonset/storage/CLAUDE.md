# Package: storage

## Purpose
Provides REST storage implementation for DaemonSet resources with support for the status subresource.

## Key Types

- **REST**: Main REST storage embedding `genericregistry.Store` for DaemonSet CRUD operations
- **StatusREST**: REST endpoint for the /status subresource

## Key Functions

- **NewREST(optsGetter)**: Creates REST and StatusREST instances with configured strategies
- **ShortNames()**: Returns ["ds"] for kubectl short name support
- **Categories()**: Returns ["all"] - DaemonSets appear in `kubectl get all`
- **StatusREST.Get()**: Retrieves DaemonSet for Patch support
- **StatusREST.Update()**: Updates only the status subset

## Design Notes

- Implements ShortNamesProvider ("ds") and CategoriesProvider ("all")
- Status subresource uses separate strategy that resets spec on updates
- No scale subresource (unlike Deployment, ReplicaSet, StatefulSet)
- Uses structured-merge-diff for field reset logic
