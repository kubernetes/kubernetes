# Change History

## Breaking Changes

### Removed Constants

1. RuleCollectionType.RuleCollectionTypeRuleCollectionTypeFirewallPolicyFilterRuleCollection
1. RuleCollectionType.RuleCollectionTypeRuleCollectionTypeFirewallPolicyNatRuleCollection
1. RuleCollectionType.RuleCollectionTypeRuleCollectionTypeFirewallPolicyRuleCollection
1. RuleType.RuleTypeRuleTypeApplicationRule
1. RuleType.RuleTypeRuleTypeFirewallPolicyRule
1. RuleType.RuleTypeRuleTypeNatRule
1. RuleType.RuleTypeRuleTypeNetworkRule

## Additive Changes

### New Constants

1. InterfaceMigrationPhase.InterfaceMigrationPhaseAbort
1. InterfaceMigrationPhase.InterfaceMigrationPhaseCommit
1. InterfaceMigrationPhase.InterfaceMigrationPhaseCommitted
1. InterfaceMigrationPhase.InterfaceMigrationPhaseNone
1. InterfaceMigrationPhase.InterfaceMigrationPhasePrepare
1. InterfaceNicType.InterfaceNicTypeElastic
1. InterfaceNicType.InterfaceNicTypeStandard
1. PublicIPAddressMigrationPhase.PublicIPAddressMigrationPhaseAbort
1. PublicIPAddressMigrationPhase.PublicIPAddressMigrationPhaseCommit
1. PublicIPAddressMigrationPhase.PublicIPAddressMigrationPhaseCommitted
1. PublicIPAddressMigrationPhase.PublicIPAddressMigrationPhaseNone
1. PublicIPAddressMigrationPhase.PublicIPAddressMigrationPhasePrepare
1. RuleCollectionType.RuleCollectionTypeFirewallPolicyFilterRuleCollection
1. RuleCollectionType.RuleCollectionTypeFirewallPolicyNatRuleCollection
1. RuleCollectionType.RuleCollectionTypeFirewallPolicyRuleCollection
1. RuleType.RuleTypeApplicationRule
1. RuleType.RuleTypeFirewallPolicyRule
1. RuleType.RuleTypeNatRule
1. RuleType.RuleTypeNetworkRule

### New Funcs

1. ApplicationSecurityGroupPropertiesFormat.MarshalJSON() ([]byte, error)
1. AzureFirewallFqdnTagPropertiesFormat.MarshalJSON() ([]byte, error)
1. AzureFirewallIPGroups.MarshalJSON() ([]byte, error)
1. AzureWebCategoryPropertiesFormat.MarshalJSON() ([]byte, error)
1. BastionActiveSession.MarshalJSON() ([]byte, error)
1. BastionSessionState.MarshalJSON() ([]byte, error)
1. BgpPeerStatus.MarshalJSON() ([]byte, error)
1. ConfigurationDiagnosticResponse.MarshalJSON() ([]byte, error)
1. ConnectivityHop.MarshalJSON() ([]byte, error)
1. ConnectivityInformation.MarshalJSON() ([]byte, error)
1. ConnectivityIssue.MarshalJSON() ([]byte, error)
1. ContainerNetworkInterfaceIPConfigurationPropertiesFormat.MarshalJSON() ([]byte, error)
1. DdosProtectionPlanPropertiesFormat.MarshalJSON() ([]byte, error)
1. ExpressRouteConnectionID.MarshalJSON() ([]byte, error)
1. ExpressRoutePortsLocationBandwidths.MarshalJSON() ([]byte, error)
1. GatewayRoute.MarshalJSON() ([]byte, error)
1. HopLinkProperties.MarshalJSON() ([]byte, error)
1. InterfaceIPConfigurationPrivateLinkConnectionProperties.MarshalJSON() ([]byte, error)
1. ManagedServiceIdentityUserAssignedIdentitiesValue.MarshalJSON() ([]byte, error)
1. PeerRoute.MarshalJSON() ([]byte, error)
1. PossibleInterfaceMigrationPhaseValues() []InterfaceMigrationPhase
1. PossibleInterfaceNicTypeValues() []InterfaceNicType
1. PossiblePublicIPAddressMigrationPhaseValues() []PublicIPAddressMigrationPhase
1. ServiceTagInformation.MarshalJSON() ([]byte, error)
1. ServiceTagInformationPropertiesFormat.MarshalJSON() ([]byte, error)
1. ServiceTagsListResult.MarshalJSON() ([]byte, error)
1. TunnelConnectionHealth.MarshalJSON() ([]byte, error)
1. VirtualApplianceNicProperties.MarshalJSON() ([]byte, error)
1. VirtualApplianceSkuInstances.MarshalJSON() ([]byte, error)
1. VirtualNetworkUsage.MarshalJSON() ([]byte, error)
1. VirtualNetworkUsageName.MarshalJSON() ([]byte, error)
1. VpnClientConnectionHealthDetail.MarshalJSON() ([]byte, error)
1. VpnSiteID.MarshalJSON() ([]byte, error)
1. WatcherPropertiesFormat.MarshalJSON() ([]byte, error)
1. WebApplicationFirewallPolicyListResult.MarshalJSON() ([]byte, error)

### Struct Changes

#### New Struct Fields

1. AvailablePrivateEndpointType.DisplayName
1. Delegation.Type
1. IPAddressAvailabilityResult.IsPlatformReserved
1. InterfaceIPConfiguration.Type
1. InterfacePropertiesFormat.MigrationPhase
1. InterfacePropertiesFormat.NicType
1. InterfacePropertiesFormat.PrivateLinkService
1. PublicIPAddressPropertiesFormat.LinkedPublicIPAddress
1. PublicIPAddressPropertiesFormat.MigrationPhase
1. PublicIPAddressPropertiesFormat.NatGateway
1. PublicIPAddressPropertiesFormat.ServicePublicIPAddress
1. PublicIPPrefixPropertiesFormat.NatGateway
1. ServiceTagInformationPropertiesFormat.State
1. Subnet.Type
1. SubnetPropertiesFormat.ApplicationGatewayIPConfigurations
1. VirtualNetworkPeering.Type
1. VirtualNetworkPeeringPropertiesFormat.DoNotVerifyRemoteGateways
1. VirtualNetworkPeeringPropertiesFormat.ResourceGUID
