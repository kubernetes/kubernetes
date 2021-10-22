# Change History

## Breaking Changes

### Removed Constants

1. CreatedByType.Application
1. CreatedByType.Key
1. CreatedByType.ManagedIdentity
1. CreatedByType.User
1. LiftrResourceCategories.MonitorLogs
1. LiftrResourceCategories.Unknown
1. ManagedIdentityTypes.SystemAssigned
1. ManagedIdentityTypes.UserAssigned
1. MarketplaceSubscriptionStatus.Active
1. MarketplaceSubscriptionStatus.Provisioning
1. MarketplaceSubscriptionStatus.Suspended
1. MarketplaceSubscriptionStatus.Unsubscribed
1. MonitoringStatus.Disabled
1. MonitoringStatus.Enabled
1. ProvisioningState.Accepted
1. ProvisioningState.Canceled
1. ProvisioningState.Creating
1. ProvisioningState.Deleted
1. ProvisioningState.Deleting
1. ProvisioningState.Failed
1. ProvisioningState.NotSpecified
1. ProvisioningState.Succeeded
1. ProvisioningState.Updating
1. SingleSignOnStates.Disable
1. SingleSignOnStates.Enable
1. SingleSignOnStates.Existing
1. SingleSignOnStates.Initial
1. TagAction.Exclude
1. TagAction.Include

### Signature Changes

#### Funcs

1. MonitorsClient.Update
	- Returns
		- From: MonitorResource, error
		- To: MonitorsUpdateFuture, error
1. MonitorsClient.UpdateSender
	- Returns
		- From: *http.Response, error
		- To: MonitorsUpdateFuture, error

## Additive Changes

### New Constants

1. CreatedByType.CreatedByTypeApplication
1. CreatedByType.CreatedByTypeKey
1. CreatedByType.CreatedByTypeManagedIdentity
1. CreatedByType.CreatedByTypeUser
1. LiftrResourceCategories.LiftrResourceCategoriesMonitorLogs
1. LiftrResourceCategories.LiftrResourceCategoriesUnknown
1. ManagedIdentityTypes.ManagedIdentityTypesSystemAssigned
1. ManagedIdentityTypes.ManagedIdentityTypesUserAssigned
1. MarketplaceSubscriptionStatus.MarketplaceSubscriptionStatusActive
1. MarketplaceSubscriptionStatus.MarketplaceSubscriptionStatusProvisioning
1. MarketplaceSubscriptionStatus.MarketplaceSubscriptionStatusSuspended
1. MarketplaceSubscriptionStatus.MarketplaceSubscriptionStatusUnsubscribed
1. MonitoringStatus.MonitoringStatusDisabled
1. MonitoringStatus.MonitoringStatusEnabled
1. ProvisioningState.ProvisioningStateAccepted
1. ProvisioningState.ProvisioningStateCanceled
1. ProvisioningState.ProvisioningStateCreating
1. ProvisioningState.ProvisioningStateDeleted
1. ProvisioningState.ProvisioningStateDeleting
1. ProvisioningState.ProvisioningStateFailed
1. ProvisioningState.ProvisioningStateNotSpecified
1. ProvisioningState.ProvisioningStateSucceeded
1. ProvisioningState.ProvisioningStateUpdating
1. SingleSignOnStates.SingleSignOnStatesDisable
1. SingleSignOnStates.SingleSignOnStatesEnable
1. SingleSignOnStates.SingleSignOnStatesExisting
1. SingleSignOnStates.SingleSignOnStatesInitial
1. TagAction.TagActionExclude
1. TagAction.TagActionInclude

### New Funcs

1. *MonitorsUpdateFuture.UnmarshalJSON([]byte) error
1. ErrorAdditionalInfo.MarshalJSON() ([]byte, error)
1. ErrorDetail.MarshalJSON() ([]byte, error)

### Struct Changes

#### New Structs

1. MonitorsUpdateFuture

#### New Struct Fields

1. MonitorResourceUpdateParameters.Sku
