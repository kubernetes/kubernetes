# Package: storage

## Purpose
Provides REST storage implementation for HorizontalPodAutoscaler (HPA) resources with status subresource.

## Key Types

- **REST**: Main REST storage embedding `genericregistry.Store` for HPA CRUD operations
- **StatusREST**: REST endpoint for the /status subresource

## Key Functions

- **NewREST(optsGetter)**: Creates REST and StatusREST instances with configured strategies
- **ShortNames()**: Returns ["hpa"] for kubectl short name support
- **Categories()**: Returns ["all"] - HPAs appear in `kubectl get all`
- **StatusREST.Get()**: Retrieves HPA for Patch support
- **StatusREST.Update()**: Updates only the status subset
- **StatusREST.GetResetFields()**: Returns fields to reset during status updates

## Design Notes

- Implements ShortNamesProvider ("hpa") and CategoriesProvider ("all")
- Status subresource uses separate strategy that resets spec on updates
- No scale subresource (HPA is itself a scaling controller, not a scalable target)
- Uses the horizontalpodautoscaler strategy package for validation
- Supports both autoscaling/v1 and autoscaling/v2 API versions
