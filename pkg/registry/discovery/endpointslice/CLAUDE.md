# Package: endpointslice

Implements the API server registry strategy for EndpointSlice resources in the `discovery.k8s.io` API group.

## Key Types

- **endpointSliceStrategy**: Implements `rest.RESTCreateStrategy`, `rest.RESTUpdateStrategy`, and `rest.RESTDeleteStrategy` for EndpointSlice objects.

## Key Functions

- **PrepareForCreate**: Initializes generation to 1 and drops disabled feature fields (topology hints).
- **PrepareForUpdate**: Increments generation on spec/label changes, handles disabled field dropping.
- **Validate / ValidateUpdate**: Validates EndpointSlice objects using declarative validation with migration checks.
- **WarningsOnCreate / WarningsOnUpdate**: Returns warnings for deprecated FQDN address types and invalid IP formats.
- **dropTopologyOnV1**: Handles v1beta1 to v1 migration by converting deprecated topology fields to nodeName.

## Design Notes

- EndpointSlices are namespace-scoped resources.
- The strategy handles feature-gated fields: `TopologyAwareHints` and `PreferSameTrafficDistribution`.
- Special handling exists for the deprecated `DeprecatedTopology` field during v1 API requests.
- Warnings are suppressed for EndpointSlices managed by known controllers (endpointslice, endpointslicemirroring).
