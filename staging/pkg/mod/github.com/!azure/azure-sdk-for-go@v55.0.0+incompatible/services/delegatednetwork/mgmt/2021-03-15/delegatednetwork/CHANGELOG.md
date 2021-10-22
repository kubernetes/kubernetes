# Change History

## Breaking Changes

### Removed Constants

1. ActionType.Internal
1. ControllerState.Deleting
1. ControllerState.Failed
1. ControllerState.Provisioning
1. ControllerState.Succeeded
1. Origin.System
1. Origin.User
1. Origin.Usersystem
1. ResourceIdentityType.None
1. ResourceIdentityType.SystemAssigned

### Removed Funcs

1. *DelegatedController.UnmarshalJSON([]byte) error
1. *DelegatedSubnet.UnmarshalJSON([]byte) error
1. *Orchestrator.UnmarshalJSON([]byte) error

### Struct Changes

#### Removed Struct Fields

1. DelegatedController.*DelegatedControllerProperties
1. DelegatedSubnet.*DelegatedSubnetProperties
1. Orchestrator.*OrchestratorResourceProperties

## Additive Changes

### New Constants

1. ActionType.ActionTypeInternal
1. ControllerState.ControllerStateDeleting
1. ControllerState.ControllerStateFailed
1. ControllerState.ControllerStateProvisioning
1. ControllerState.ControllerStateSucceeded
1. Origin.OriginSystem
1. Origin.OriginUser
1. Origin.OriginUsersystem
1. ResourceIdentityType.ResourceIdentityTypeNone
1. ResourceIdentityType.ResourceIdentityTypeSystemAssigned

### New Funcs

1. DelegatedControllerProperties.MarshalJSON() ([]byte, error)
1. ErrorAdditionalInfo.MarshalJSON() ([]byte, error)
1. ErrorDetail.MarshalJSON() ([]byte, error)
1. OperationDisplay.MarshalJSON() ([]byte, error)
1. OperationListResult.MarshalJSON() ([]byte, error)

### Struct Changes

#### New Struct Fields

1. DelegatedController.Properties
1. DelegatedSubnet.Properties
1. Orchestrator.Properties
