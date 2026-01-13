# Package: flowcontrol

## Purpose
Internal (unversioned) API types for the flowcontrol.apiserver.k8s.io API group, implementing API Priority and Fairness (APF) for request handling.

## Key Types

- **FlowSchema**: Defines how to classify and handle API requests. Matches requests to priority levels based on rules.
- **FlowSchemaSpec**: MatchingPrecedence, PriorityLevelConfiguration reference, distinguisher method, and rules.
- **PriorityLevelConfiguration**: Defines resource limits and queuing behavior for a priority level.
- **PriorityLevelConfigurationSpec**: Type (Exempt/Limited), limited config with queuing parameters.

## Key Constants

### Wildcards
- **APIGroupAll, ResourceAll, VerbAll, NonResourceAll, NameAll**: "*" wildcards.
- **NamespaceEvery**: "*" for namespace matching.

### System Priority Levels
- **PriorityLevelConfigurationNameExempt**: "exempt" - bypasses flow control.
- **PriorityLevelConfigurationNameCatchAll**: "catch-all" - default priority level.
- **FlowSchemaNameExempt, FlowSchemaNameCatchAll**: Corresponding flow schemas.

### Validation
- **FlowSchemaMaxMatchingPrecedence**: 10000 - maximum precedence value.

## Key Functions

- **Kind(kind string)**: Returns Group-qualified GroupKind.
- **Resource(resource string)**: Returns Group-qualified GroupResource.
- **AddToScheme**: Registers FlowSchema and PriorityLevelConfiguration types.

## Design Notes

- Implements fair queuing to prevent any single client from monopolizing the API server.
- Flow distinguishers can be by namespace or user.
- Limited priority levels support queuing with configurable queue count, hand size, and queue length.
