# Package: service

## Purpose
Provides the registry interface and REST strategy implementation for storing Service API objects, including proxy support and complex type-dependent field handling.

## Key Types

- **svcStrategy**: Main strategy for Service CRUD operations (namespace-scoped).
- **serviceStatusStrategy**: Strategy for /status subresource updates.
- **ProxyREST**: Implements /proxy endpoint for HTTP proxying to services.

## Key Functions

- **Strategy** (var): Default logic for creating/updating Services.
- **StatusStrategy** (var): Strategy for status subresource.
- **NamespaceScoped()**: Returns `true` - Services are namespace-scoped.
- **PrepareForCreate()**: Clears status, drops disabled fields.
- **PrepareForUpdate()**: Preserves status, drops type-dependent fields when service type changes.
- **dropTypeDependentFields()**: Intelligently clears ClusterIP, NodePort, HealthCheckNodePort, LoadBalancerClass when switching service types.
- **SelectableFields()**: Returns filterable fields including spec.clusterIP and spec.type.
- **ProxyREST.Connect()**: Returns handler for proxying HTTP to service endpoints.

## Design Notes

- Services are a discriminated union where `type` is the discriminator.
- Automatically clears fields that don't apply to new service type on update.
- Feature-gated: ServiceTrafficDistribution.
- Field indexing on spec.clusterIP and spec.type for efficient queries.
- Complex logic to handle ClusterIP/ClusterIPs, NodePorts, HealthCheckNodePort during type transitions.
