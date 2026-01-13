# Package: storage

## Purpose
Provides REST storage implementation for Job resources with status subresource and delete warnings.

## Key Types

- **JobStorage**: Container struct with Job and Status REST endpoints
- **REST**: Main REST storage embedding `genericregistry.Store` for Job CRUD operations
- **StatusREST**: REST endpoint for the /status subresource

## Key Functions

- **NewStorage(optsGetter)**: Creates JobStorage with Job and StatusREST
- **NewREST(optsGetter)**: Creates REST and StatusREST instances with configured strategies
- **Categories()**: Returns ["all"] - Jobs appear in `kubectl get all`
- **Delete()**: Warns if propagationPolicy not set (default orphans pods in v1)
- **DeleteCollection()**: Same warning behavior as Delete
- **StatusREST.Get/Update()**: Standard status subresource operations

## Design Notes

- Implements CategoriesProvider ("all") but NOT ShortNamesProvider (no kubectl shortname)
- Custom Delete/DeleteCollection that warn about orphaning pods
- Warning: "child pods are preserved by default when jobs are deleted"
- Uses PredicateFunc for efficient watch filtering
- Status subresource uses separate strategy with strengthened validation
