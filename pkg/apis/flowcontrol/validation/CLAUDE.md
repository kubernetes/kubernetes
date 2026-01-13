# Package: validation

## Purpose
Provides validation logic for FlowSchema and PriorityLevelConfiguration resources.

## Key Functions

### FlowSchema Validation
- **ValidateFlowSchema(fs)**: Full validation including metadata, spec, status.
- **ValidateFlowSchemaUpdate(old, fs)**: Update validation.
- **ValidateFlowSchemaSpec(name, spec, fldPath)**: Validates spec fields.

### PriorityLevelConfiguration Validation
- **ValidatePriorityLevelConfiguration(pl, opts)**: Full validation.
- **ValidatePriorityLevelConfigurationUpdate(old, pl, opts)**: Update validation.
- **ValidatePriorityLevelConfigurationSpec(spec, fldPath, opts)**: Validates spec fields.

## Validation Rules

### FlowSchema
- MatchingPrecedence: 1-10000
- PriorityLevelConfiguration reference required
- DistinguisherMethod: ByNamespace or ByUser
- Rules must have valid subjects (User, Group, ServiceAccount) and resource/non-resource rules
- Mandatory FlowSchemas must match their bootstrap spec exactly

### PriorityLevelConfiguration
- Type: Exempt or Limited
- Limited config requires valid queuing parameters
- Queuing: up to 10^7 queues, valid hand size, queue length
- Mandatory priority levels must match bootstrap spec

## Supported Values

- **Verbs**: get, list, create, update, delete, deletecollection, patch, watch, proxy
- **Subject kinds**: ServiceAccount, Group, User
- **Limit response types**: Queue, Reject
