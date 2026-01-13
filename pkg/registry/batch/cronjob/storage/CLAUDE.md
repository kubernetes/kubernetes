# Package: storage

## Purpose
Provides REST storage implementation for CronJob resources with status subresource.

## Key Types

- **REST**: Main REST storage embedding `genericregistry.Store` for CronJob CRUD operations
- **StatusREST**: REST endpoint for the /status subresource

## Key Functions

- **NewREST(optsGetter)**: Creates REST and StatusREST instances with configured strategies
- **ShortNames()**: Returns ["cj"] for kubectl short name support
- **Categories()**: Returns ["all"] - CronJobs appear in `kubectl get all`
- **StatusREST.Get()**: Retrieves CronJob for Patch support
- **StatusREST.Update()**: Updates only the status subset

## Design Notes

- Implements ShortNamesProvider ("cj") and CategoriesProvider ("all")
- Status subresource uses separate strategy that resets spec on updates
- No scale subresource (CronJobs manage job count via schedule, not replicas)
- Uses the cronjob strategy package for validation
