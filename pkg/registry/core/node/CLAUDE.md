# Package: node

## Purpose
Provides the registry interface and REST strategy implementation for storing Node API objects, including status subresource handling and proxy functionality.

## Key Types

- **nodeStrategy**: Main strategy for Node CRUD operations (cluster-scoped).
- **nodeStatusStrategy**: Strategy for /status subresource updates.
- **ResourceGetter**: Interface for retrieving resources by ResourceLocation.

## Key Functions

- **Strategy** (var): Default logic for creating/updating Nodes.
- **StatusStrategy** (var): Strategy for status subresource.
- **NamespaceScoped()**: Returns `false` - Nodes are cluster-scoped.
- **GetResetFields()**: Returns fields to reset (status for main, spec for status strategy).
- **PrepareForCreate/Update()**: Handles disabled fields (ConfigSource, RuntimeHandlers, Features, DeclaredFeatures) based on feature gates.
- **dropDisabledFields()**: Conditionally drops fields based on feature gates and previous usage.
- **NodeToSelectableFields()**: Returns filterable fields including `spec.unschedulable`.
- **GetAttrs()**: Returns labels and fields for filtering.
- **MatchNode()**: Returns selection predicate for filtering.
- **ResourceLocation()**: Returns URL and transport for proxying to a node's kubelet.
- **isProxyableHostname()**: Security check ensuring hostname resolves to global unicast addresses.
- **nodeWarnings()**: Generates warnings for deprecated fields (configSource, externalID, podCIDRs).

## Design Notes

- Cluster-scoped resource.
- Implements field reset strategies for server-side apply.
- Feature-gate controlled fields: RuntimeHandlers, Features, DeclaredFeatures.
- Proxy support with hostname validation for security.
- Warnings for deprecated spec.configSource and spec.externalID.
