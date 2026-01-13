# Package: endpoint

## Purpose
Provides the registry interface and REST strategy implementation for storing Endpoints API objects in the Kubernetes API server.

## Key Types

- **endpointsStrategy**: Implements REST create/update strategies for Endpoints. Embeds `runtime.ObjectTyper` and `names.NameGenerator`.

## Key Functions

- **Strategy** (var): Default logic for creating/updating Endpoints via REST API.
- **NamespaceScoped()**: Returns `true` - Endpoints are namespace-scoped.
- **AllowCreateOnUpdate()**: Returns `true` - Endpoints can be created via update operations.
- **Validate()**: Validates new Endpoints using `ValidateEndpointsCreate`.
- **ValidateUpdate()**: Validates Endpoint updates.
- **WarningsOnCreate/Update()**: Returns warnings for IP validation issues (skipped for controller-managed endpoints).
- **endpointsWarnings()**: Helper that checks for bad IPs in endpoint addresses, skipping validation for endpoints managed by the endpoints controller.

## Design Notes

- Allows unconditional updates and create-on-update semantics.
- Optimizes warning generation by skipping IP checks for controller-managed endpoints (detected via `endpointscontroller.LabelManagedBy` label).
- Validates both ready and not-ready addresses for IP format warnings.
