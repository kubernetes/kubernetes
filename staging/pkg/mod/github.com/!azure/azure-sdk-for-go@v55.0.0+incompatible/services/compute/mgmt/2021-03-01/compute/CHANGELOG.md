# Change History

## Breaking Changes

### Removed Constants

1. AccessLevel.None
1. AccessLevel.Read
1. AccessLevel.Write
1. AggregatedReplicationState.Completed
1. AggregatedReplicationState.Failed
1. AggregatedReplicationState.InProgress
1. AggregatedReplicationState.Unknown
1. AvailabilitySetSkuTypes.Aligned
1. AvailabilitySetSkuTypes.Classic
1. CloudServiceUpgradeMode.Auto
1. CloudServiceUpgradeMode.Manual
1. CloudServiceUpgradeMode.Simultaneous
1. ComponentNames.MicrosoftWindowsShellSetup
1. DiffDiskOptions.Local
1. DiffDiskPlacement.CacheDisk
1. DiffDiskPlacement.ResourceDisk
1. DiskCreateOption.Attach
1. DiskCreateOption.Copy
1. DiskCreateOption.Empty
1. DiskCreateOption.FromImage
1. DiskCreateOption.Import
1. DiskCreateOption.Restore
1. DiskCreateOption.Upload
1. DiskDetachOptionTypes.ForceDetach
1. DiskEncryptionSetType.EncryptionAtRestWithCustomerKey
1. DiskEncryptionSetType.EncryptionAtRestWithPlatformAndCustomerKeys
1. DiskSecurityTypes.TrustedLaunch
1. DiskState.ActiveSAS
1. DiskState.ActiveUpload
1. DiskState.Attached
1. DiskState.ReadyToUpload
1. DiskState.Reserved
1. DiskState.Unattached
1. DiskStorageAccountTypes.PremiumLRS
1. DiskStorageAccountTypes.PremiumZRS
1. DiskStorageAccountTypes.StandardLRS
1. DiskStorageAccountTypes.StandardSSDLRS
1. DiskStorageAccountTypes.StandardSSDZRS
1. DiskStorageAccountTypes.UltraSSDLRS
1. ExtendedLocationTypes.EdgeZone
1. HyperVGeneration.V1
1. HyperVGeneration.V2
1. IPVersion.IPv4
1. IPVersion.IPv6
1. InstanceViewTypes.InstanceView
1. IntervalInMins.FiveMins
1. IntervalInMins.SixtyMins
1. IntervalInMins.ThirtyMins
1. IntervalInMins.ThreeMins
1. LinuxVMGuestPatchMode.AutomaticByPlatform
1. LinuxVMGuestPatchMode.ImageDefault
1. NetworkAccessPolicy.AllowAll
1. NetworkAccessPolicy.AllowPrivate
1. NetworkAccessPolicy.DenyAll
1. OperatingSystemStateTypes.Generalized
1. OperatingSystemStateTypes.Specialized
1. OperatingSystemTypes.Linux
1. OperatingSystemTypes.Windows
1. OrchestrationMode.Flexible
1. OrchestrationMode.Uniform
1. OrchestrationServiceNames.AutomaticRepairs
1. OrchestrationServiceState.NotRunning
1. OrchestrationServiceState.Running
1. OrchestrationServiceState.Suspended
1. OrchestrationServiceStateAction.Resume
1. OrchestrationServiceStateAction.Suspend
1. PassNames.OobeSystem
1. PrivateEndpointServiceConnectionStatus.Approved
1. PrivateEndpointServiceConnectionStatus.Pending
1. PrivateEndpointServiceConnectionStatus.Rejected
1. ProtocolTypes.HTTP
1. ProtocolTypes.HTTPS
1. ProximityPlacementGroupType.Standard
1. ProximityPlacementGroupType.Ultra
1. ResourceSkuRestrictionsReasonCode.NotAvailableForSubscription
1. ResourceSkuRestrictionsReasonCode.QuotaID
1. ResourceSkuRestrictionsType.Location
1. ResourceSkuRestrictionsType.Zone
1. RollingUpgradeActionType.Cancel
1. RollingUpgradeActionType.Start
1. SettingNames.AutoLogon
1. SettingNames.FirstLogonCommands
1. StatusLevelTypes.Error
1. StatusLevelTypes.Info
1. StatusLevelTypes.Warning
1. VMGuestPatchClassificationLinux.Critical
1. VMGuestPatchClassificationLinux.Other
1. VMGuestPatchClassificationLinux.Security
1. VMGuestPatchRebootSetting.Always
1. VMGuestPatchRebootSetting.IfRequired
1. VMGuestPatchRebootSetting.Never
1. VirtualMachineEvictionPolicyTypes.Deallocate
1. VirtualMachineEvictionPolicyTypes.Delete
1. VirtualMachinePriorityTypes.Low
1. VirtualMachinePriorityTypes.Regular
1. VirtualMachinePriorityTypes.Spot
1. VirtualMachineScaleSetScaleInRules.Default
1. VirtualMachineScaleSetScaleInRules.NewestVM
1. VirtualMachineScaleSetScaleInRules.OldestVM
1. VirtualMachineSizeTypes.BasicA0
1. VirtualMachineSizeTypes.BasicA1
1. VirtualMachineSizeTypes.BasicA2
1. VirtualMachineSizeTypes.BasicA3
1. VirtualMachineSizeTypes.BasicA4
1. VirtualMachineSizeTypes.StandardA0
1. VirtualMachineSizeTypes.StandardA1
1. VirtualMachineSizeTypes.StandardA10
1. VirtualMachineSizeTypes.StandardA11
1. VirtualMachineSizeTypes.StandardA1V2
1. VirtualMachineSizeTypes.StandardA2
1. VirtualMachineSizeTypes.StandardA2V2
1. VirtualMachineSizeTypes.StandardA2mV2
1. VirtualMachineSizeTypes.StandardA3
1. VirtualMachineSizeTypes.StandardA4
1. VirtualMachineSizeTypes.StandardA4V2
1. VirtualMachineSizeTypes.StandardA4mV2
1. VirtualMachineSizeTypes.StandardA5
1. VirtualMachineSizeTypes.StandardA6
1. VirtualMachineSizeTypes.StandardA7
1. VirtualMachineSizeTypes.StandardA8
1. VirtualMachineSizeTypes.StandardA8V2
1. VirtualMachineSizeTypes.StandardA8mV2
1. VirtualMachineSizeTypes.StandardA9
1. VirtualMachineSizeTypes.StandardB1ms
1. VirtualMachineSizeTypes.StandardB1s
1. VirtualMachineSizeTypes.StandardB2ms
1. VirtualMachineSizeTypes.StandardB2s
1. VirtualMachineSizeTypes.StandardB4ms
1. VirtualMachineSizeTypes.StandardB8ms
1. VirtualMachineSizeTypes.StandardD1
1. VirtualMachineSizeTypes.StandardD11
1. VirtualMachineSizeTypes.StandardD11V2
1. VirtualMachineSizeTypes.StandardD12
1. VirtualMachineSizeTypes.StandardD12V2
1. VirtualMachineSizeTypes.StandardD13
1. VirtualMachineSizeTypes.StandardD13V2
1. VirtualMachineSizeTypes.StandardD14
1. VirtualMachineSizeTypes.StandardD14V2
1. VirtualMachineSizeTypes.StandardD15V2
1. VirtualMachineSizeTypes.StandardD16V3
1. VirtualMachineSizeTypes.StandardD16sV3
1. VirtualMachineSizeTypes.StandardD1V2
1. VirtualMachineSizeTypes.StandardD2
1. VirtualMachineSizeTypes.StandardD2V2
1. VirtualMachineSizeTypes.StandardD2V3
1. VirtualMachineSizeTypes.StandardD2sV3
1. VirtualMachineSizeTypes.StandardD3
1. VirtualMachineSizeTypes.StandardD32V3
1. VirtualMachineSizeTypes.StandardD32sV3
1. VirtualMachineSizeTypes.StandardD3V2
1. VirtualMachineSizeTypes.StandardD4
1. VirtualMachineSizeTypes.StandardD4V2
1. VirtualMachineSizeTypes.StandardD4V3
1. VirtualMachineSizeTypes.StandardD4sV3
1. VirtualMachineSizeTypes.StandardD5V2
1. VirtualMachineSizeTypes.StandardD64V3
1. VirtualMachineSizeTypes.StandardD64sV3
1. VirtualMachineSizeTypes.StandardD8V3
1. VirtualMachineSizeTypes.StandardD8sV3
1. VirtualMachineSizeTypes.StandardDS1
1. VirtualMachineSizeTypes.StandardDS11
1. VirtualMachineSizeTypes.StandardDS11V2
1. VirtualMachineSizeTypes.StandardDS12
1. VirtualMachineSizeTypes.StandardDS12V2
1. VirtualMachineSizeTypes.StandardDS13
1. VirtualMachineSizeTypes.StandardDS132V2
1. VirtualMachineSizeTypes.StandardDS134V2
1. VirtualMachineSizeTypes.StandardDS13V2
1. VirtualMachineSizeTypes.StandardDS14
1. VirtualMachineSizeTypes.StandardDS144V2
1. VirtualMachineSizeTypes.StandardDS148V2
1. VirtualMachineSizeTypes.StandardDS14V2
1. VirtualMachineSizeTypes.StandardDS15V2
1. VirtualMachineSizeTypes.StandardDS1V2
1. VirtualMachineSizeTypes.StandardDS2
1. VirtualMachineSizeTypes.StandardDS2V2
1. VirtualMachineSizeTypes.StandardDS3
1. VirtualMachineSizeTypes.StandardDS3V2
1. VirtualMachineSizeTypes.StandardDS4
1. VirtualMachineSizeTypes.StandardDS4V2
1. VirtualMachineSizeTypes.StandardDS5V2
1. VirtualMachineSizeTypes.StandardE16V3
1. VirtualMachineSizeTypes.StandardE16sV3
1. VirtualMachineSizeTypes.StandardE2V3
1. VirtualMachineSizeTypes.StandardE2sV3
1. VirtualMachineSizeTypes.StandardE3216V3
1. VirtualMachineSizeTypes.StandardE328sV3
1. VirtualMachineSizeTypes.StandardE32V3
1. VirtualMachineSizeTypes.StandardE32sV3
1. VirtualMachineSizeTypes.StandardE4V3
1. VirtualMachineSizeTypes.StandardE4sV3
1. VirtualMachineSizeTypes.StandardE6416sV3
1. VirtualMachineSizeTypes.StandardE6432sV3
1. VirtualMachineSizeTypes.StandardE64V3
1. VirtualMachineSizeTypes.StandardE64sV3
1. VirtualMachineSizeTypes.StandardE8V3
1. VirtualMachineSizeTypes.StandardE8sV3
1. VirtualMachineSizeTypes.StandardF1
1. VirtualMachineSizeTypes.StandardF16
1. VirtualMachineSizeTypes.StandardF16s
1. VirtualMachineSizeTypes.StandardF16sV2
1. VirtualMachineSizeTypes.StandardF1s
1. VirtualMachineSizeTypes.StandardF2
1. VirtualMachineSizeTypes.StandardF2s
1. VirtualMachineSizeTypes.StandardF2sV2
1. VirtualMachineSizeTypes.StandardF32sV2
1. VirtualMachineSizeTypes.StandardF4
1. VirtualMachineSizeTypes.StandardF4s
1. VirtualMachineSizeTypes.StandardF4sV2
1. VirtualMachineSizeTypes.StandardF64sV2
1. VirtualMachineSizeTypes.StandardF72sV2
1. VirtualMachineSizeTypes.StandardF8
1. VirtualMachineSizeTypes.StandardF8s
1. VirtualMachineSizeTypes.StandardF8sV2
1. VirtualMachineSizeTypes.StandardG1
1. VirtualMachineSizeTypes.StandardG2
1. VirtualMachineSizeTypes.StandardG3
1. VirtualMachineSizeTypes.StandardG4
1. VirtualMachineSizeTypes.StandardG5
1. VirtualMachineSizeTypes.StandardGS1
1. VirtualMachineSizeTypes.StandardGS2
1. VirtualMachineSizeTypes.StandardGS3
1. VirtualMachineSizeTypes.StandardGS4
1. VirtualMachineSizeTypes.StandardGS44
1. VirtualMachineSizeTypes.StandardGS48
1. VirtualMachineSizeTypes.StandardGS5
1. VirtualMachineSizeTypes.StandardGS516
1. VirtualMachineSizeTypes.StandardGS58
1. VirtualMachineSizeTypes.StandardH16
1. VirtualMachineSizeTypes.StandardH16m
1. VirtualMachineSizeTypes.StandardH16mr
1. VirtualMachineSizeTypes.StandardH16r
1. VirtualMachineSizeTypes.StandardH8
1. VirtualMachineSizeTypes.StandardH8m
1. VirtualMachineSizeTypes.StandardL16s
1. VirtualMachineSizeTypes.StandardL32s
1. VirtualMachineSizeTypes.StandardL4s
1. VirtualMachineSizeTypes.StandardL8s
1. VirtualMachineSizeTypes.StandardM12832ms
1. VirtualMachineSizeTypes.StandardM12864ms
1. VirtualMachineSizeTypes.StandardM128ms
1. VirtualMachineSizeTypes.StandardM128s
1. VirtualMachineSizeTypes.StandardM6416ms
1. VirtualMachineSizeTypes.StandardM6432ms
1. VirtualMachineSizeTypes.StandardM64ms
1. VirtualMachineSizeTypes.StandardM64s
1. VirtualMachineSizeTypes.StandardNC12
1. VirtualMachineSizeTypes.StandardNC12sV2
1. VirtualMachineSizeTypes.StandardNC12sV3
1. VirtualMachineSizeTypes.StandardNC24
1. VirtualMachineSizeTypes.StandardNC24r
1. VirtualMachineSizeTypes.StandardNC24rsV2
1. VirtualMachineSizeTypes.StandardNC24rsV3
1. VirtualMachineSizeTypes.StandardNC24sV2
1. VirtualMachineSizeTypes.StandardNC24sV3
1. VirtualMachineSizeTypes.StandardNC6
1. VirtualMachineSizeTypes.StandardNC6sV2
1. VirtualMachineSizeTypes.StandardNC6sV3
1. VirtualMachineSizeTypes.StandardND12s
1. VirtualMachineSizeTypes.StandardND24rs
1. VirtualMachineSizeTypes.StandardND24s
1. VirtualMachineSizeTypes.StandardND6s
1. VirtualMachineSizeTypes.StandardNV12
1. VirtualMachineSizeTypes.StandardNV24
1. VirtualMachineSizeTypes.StandardNV6

### Signature Changes

#### Funcs

1. GalleriesClient.Get
	- Params
		- From: context.Context, string, string
		- To: context.Context, string, string, SelectPermissions
1. GalleriesClient.GetPreparer
	- Params
		- From: context.Context, string, string
		- To: context.Context, string, string, SelectPermissions
1. VirtualMachineScaleSetsClient.Get
	- Params
		- From: context.Context, string, string
		- To: context.Context, string, string, ExpandTypesForGetVMScaleSets
1. VirtualMachineScaleSetsClient.GetPreparer
	- Params
		- From: context.Context, string, string
		- To: context.Context, string, string, ExpandTypesForGetVMScaleSets

#### Struct Fields

1. OrchestrationServiceStateInput.ServiceName changed type from OrchestrationServiceNames to *string

## Additive Changes

### New Constants

1. AccessLevel.AccessLevelNone
1. AccessLevel.AccessLevelRead
1. AccessLevel.AccessLevelWrite
1. AggregatedReplicationState.AggregatedReplicationStateCompleted
1. AggregatedReplicationState.AggregatedReplicationStateFailed
1. AggregatedReplicationState.AggregatedReplicationStateInProgress
1. AggregatedReplicationState.AggregatedReplicationStateUnknown
1. AvailabilitySetSkuTypes.AvailabilitySetSkuTypesAligned
1. AvailabilitySetSkuTypes.AvailabilitySetSkuTypesClassic
1. CloudServiceUpgradeMode.CloudServiceUpgradeModeAuto
1. CloudServiceUpgradeMode.CloudServiceUpgradeModeManual
1. CloudServiceUpgradeMode.CloudServiceUpgradeModeSimultaneous
1. ComponentNames.ComponentNamesMicrosoftWindowsShellSetup
1. ConsistencyModeTypes.ConsistencyModeTypesApplicationConsistent
1. ConsistencyModeTypes.ConsistencyModeTypesCrashConsistent
1. ConsistencyModeTypes.ConsistencyModeTypesFileSystemConsistent
1. DeleteOptions.DeleteOptionsDelete
1. DeleteOptions.DeleteOptionsDetach
1. DiffDiskOptions.DiffDiskOptionsLocal
1. DiffDiskPlacement.DiffDiskPlacementCacheDisk
1. DiffDiskPlacement.DiffDiskPlacementResourceDisk
1. DiskCreateOption.DiskCreateOptionAttach
1. DiskCreateOption.DiskCreateOptionCopy
1. DiskCreateOption.DiskCreateOptionEmpty
1. DiskCreateOption.DiskCreateOptionFromImage
1. DiskCreateOption.DiskCreateOptionImport
1. DiskCreateOption.DiskCreateOptionRestore
1. DiskCreateOption.DiskCreateOptionUpload
1. DiskDeleteOptionTypes.DiskDeleteOptionTypesDelete
1. DiskDeleteOptionTypes.DiskDeleteOptionTypesDetach
1. DiskDetachOptionTypes.DiskDetachOptionTypesForceDetach
1. DiskEncryptionSetType.DiskEncryptionSetTypeEncryptionAtRestWithCustomerKey
1. DiskEncryptionSetType.DiskEncryptionSetTypeEncryptionAtRestWithPlatformAndCustomerKeys
1. DiskSecurityTypes.DiskSecurityTypesTrustedLaunch
1. DiskState.DiskStateActiveSAS
1. DiskState.DiskStateActiveUpload
1. DiskState.DiskStateAttached
1. DiskState.DiskStateReadyToUpload
1. DiskState.DiskStateReserved
1. DiskState.DiskStateUnattached
1. DiskStorageAccountTypes.DiskStorageAccountTypesPremiumLRS
1. DiskStorageAccountTypes.DiskStorageAccountTypesPremiumZRS
1. DiskStorageAccountTypes.DiskStorageAccountTypesStandardLRS
1. DiskStorageAccountTypes.DiskStorageAccountTypesStandardSSDLRS
1. DiskStorageAccountTypes.DiskStorageAccountTypesStandardSSDZRS
1. DiskStorageAccountTypes.DiskStorageAccountTypesUltraSSDLRS
1. ExpandTypesForGetVMScaleSets.ExpandTypesForGetVMScaleSetsUserData
1. ExtendedLocationTypes.ExtendedLocationTypesEdgeZone
1. GallerySharingPermissionTypes.GallerySharingPermissionTypesGroups
1. GallerySharingPermissionTypes.GallerySharingPermissionTypesPrivate
1. HyperVGeneration.HyperVGenerationV1
1. HyperVGeneration.HyperVGenerationV2
1. IPVersion.IPVersionIPv4
1. IPVersion.IPVersionIPv6
1. IPVersions.IPVersionsIPv4
1. IPVersions.IPVersionsIPv6
1. InstanceViewTypes.InstanceViewTypesInstanceView
1. InstanceViewTypes.InstanceViewTypesUserData
1. IntervalInMins.IntervalInMinsFiveMins
1. IntervalInMins.IntervalInMinsSixtyMins
1. IntervalInMins.IntervalInMinsThirtyMins
1. IntervalInMins.IntervalInMinsThreeMins
1. LinuxPatchAssessmentMode.LinuxPatchAssessmentModeAutomaticByPlatform
1. LinuxPatchAssessmentMode.LinuxPatchAssessmentModeImageDefault
1. LinuxVMGuestPatchMode.LinuxVMGuestPatchModeAutomaticByPlatform
1. LinuxVMGuestPatchMode.LinuxVMGuestPatchModeImageDefault
1. NetworkAPIVersion.NetworkAPIVersionTwoZeroTwoZeroHyphenMinusOneOneHyphenMinusZeroOne
1. NetworkAccessPolicy.NetworkAccessPolicyAllowAll
1. NetworkAccessPolicy.NetworkAccessPolicyAllowPrivate
1. NetworkAccessPolicy.NetworkAccessPolicyDenyAll
1. OperatingSystemStateTypes.OperatingSystemStateTypesGeneralized
1. OperatingSystemStateTypes.OperatingSystemStateTypesSpecialized
1. OperatingSystemType.OperatingSystemTypeLinux
1. OperatingSystemType.OperatingSystemTypeWindows
1. OperatingSystemTypes.OperatingSystemTypesLinux
1. OperatingSystemTypes.OperatingSystemTypesWindows
1. OrchestrationMode.OrchestrationModeFlexible
1. OrchestrationMode.OrchestrationModeUniform
1. OrchestrationServiceNames.OrchestrationServiceNamesAutomaticRepairs
1. OrchestrationServiceState.OrchestrationServiceStateNotRunning
1. OrchestrationServiceState.OrchestrationServiceStateRunning
1. OrchestrationServiceState.OrchestrationServiceStateSuspended
1. OrchestrationServiceStateAction.OrchestrationServiceStateActionResume
1. OrchestrationServiceStateAction.OrchestrationServiceStateActionSuspend
1. PassNames.PassNamesOobeSystem
1. PrivateEndpointServiceConnectionStatus.PrivateEndpointServiceConnectionStatusApproved
1. PrivateEndpointServiceConnectionStatus.PrivateEndpointServiceConnectionStatusPending
1. PrivateEndpointServiceConnectionStatus.PrivateEndpointServiceConnectionStatusRejected
1. ProtocolTypes.ProtocolTypesHTTP
1. ProtocolTypes.ProtocolTypesHTTPS
1. ProximityPlacementGroupType.ProximityPlacementGroupTypeStandard
1. ProximityPlacementGroupType.ProximityPlacementGroupTypeUltra
1. PublicIPAddressSkuName.PublicIPAddressSkuNameBasic
1. PublicIPAddressSkuName.PublicIPAddressSkuNameStandard
1. PublicIPAddressSkuTier.PublicIPAddressSkuTierGlobal
1. PublicIPAddressSkuTier.PublicIPAddressSkuTierRegional
1. PublicIPAllocationMethod.PublicIPAllocationMethodDynamic
1. PublicIPAllocationMethod.PublicIPAllocationMethodStatic
1. ResourceSkuRestrictionsReasonCode.ResourceSkuRestrictionsReasonCodeNotAvailableForSubscription
1. ResourceSkuRestrictionsReasonCode.ResourceSkuRestrictionsReasonCodeQuotaID
1. ResourceSkuRestrictionsType.ResourceSkuRestrictionsTypeLocation
1. ResourceSkuRestrictionsType.ResourceSkuRestrictionsTypeZone
1. RestorePointCollectionExpandOptions.RestorePointCollectionExpandOptionsRestorePoints
1. RollingUpgradeActionType.RollingUpgradeActionTypeCancel
1. RollingUpgradeActionType.RollingUpgradeActionTypeStart
1. SelectPermissions.SelectPermissionsPermissions
1. SettingNames.SettingNamesAutoLogon
1. SettingNames.SettingNamesFirstLogonCommands
1. SharedToValues.SharedToValuesTenant
1. SharingProfileGroupTypes.SharingProfileGroupTypesAADTenants
1. SharingProfileGroupTypes.SharingProfileGroupTypesSubscriptions
1. SharingUpdateOperationTypes.SharingUpdateOperationTypesAdd
1. SharingUpdateOperationTypes.SharingUpdateOperationTypesRemove
1. SharingUpdateOperationTypes.SharingUpdateOperationTypesReset
1. StatusLevelTypes.StatusLevelTypesError
1. StatusLevelTypes.StatusLevelTypesInfo
1. StatusLevelTypes.StatusLevelTypesWarning
1. VMGuestPatchClassificationLinux.VMGuestPatchClassificationLinuxCritical
1. VMGuestPatchClassificationLinux.VMGuestPatchClassificationLinuxOther
1. VMGuestPatchClassificationLinux.VMGuestPatchClassificationLinuxSecurity
1. VMGuestPatchRebootSetting.VMGuestPatchRebootSettingAlways
1. VMGuestPatchRebootSetting.VMGuestPatchRebootSettingIfRequired
1. VMGuestPatchRebootSetting.VMGuestPatchRebootSettingNever
1. VirtualMachineEvictionPolicyTypes.VirtualMachineEvictionPolicyTypesDeallocate
1. VirtualMachineEvictionPolicyTypes.VirtualMachineEvictionPolicyTypesDelete
1. VirtualMachinePriorityTypes.VirtualMachinePriorityTypesLow
1. VirtualMachinePriorityTypes.VirtualMachinePriorityTypesRegular
1. VirtualMachinePriorityTypes.VirtualMachinePriorityTypesSpot
1. VirtualMachineScaleSetScaleInRules.VirtualMachineScaleSetScaleInRulesDefault
1. VirtualMachineScaleSetScaleInRules.VirtualMachineScaleSetScaleInRulesNewestVM
1. VirtualMachineScaleSetScaleInRules.VirtualMachineScaleSetScaleInRulesOldestVM
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesBasicA0
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesBasicA1
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesBasicA2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesBasicA3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesBasicA4
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA0
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA1
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA10
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA11
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA1V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA2V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA2mV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA4
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA4V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA4mV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA5
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA6
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA7
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA8
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA8V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA8mV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardA9
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardB1ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardB1s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardB2ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardB2s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardB4ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardB8ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD1
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD11
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD11V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD12
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD12V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD13
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD13V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD14
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD14V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD15V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD16V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD16sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD1V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD2V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD2V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD2sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD32V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD32sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD3V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD4
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD4V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD4V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD4sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD5V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD64V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD64sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD8V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardD8sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS1
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS11
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS11V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS12
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS12V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS13
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS132V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS134V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS13V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS14
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS144V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS148V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS14V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS15V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS1V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS2V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS3V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS4
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS4V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardDS5V2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE16V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE16sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE2V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE2sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE3216V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE328sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE32V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE32sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE4V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE4sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE6416sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE6432sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE64V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE64sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE8V3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardE8sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF1
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF16
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF16s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF16sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF1s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF2s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF2sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF32sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF4
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF4s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF4sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF64sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF72sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF8
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF8s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardF8sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardG1
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardG2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardG3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardG4
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardG5
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS1
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS4
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS44
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS48
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS5
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS516
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardGS58
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardH16
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardH16m
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardH16mr
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardH16r
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardH8
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardH8m
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardL16s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardL32s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardL4s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardL8s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM12832ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM12864ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM128ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM128s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM6416ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM6432ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM64ms
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardM64s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC12
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC12sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC12sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC24
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC24r
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC24rsV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC24rsV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC24sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC24sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC6
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC6sV2
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNC6sV3
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardND12s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardND24rs
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardND24s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardND6s
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNV12
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNV24
1. VirtualMachineSizeTypes.VirtualMachineSizeTypesStandardNV6
1. WindowsPatchAssessmentMode.WindowsPatchAssessmentModeAutomaticByPlatform
1. WindowsPatchAssessmentMode.WindowsPatchAssessmentModeImageDefault

### New Funcs

1. *GallerySharingProfileUpdateFuture.UnmarshalJSON([]byte) error
1. *PirSharedGalleryResource.UnmarshalJSON([]byte) error
1. *RestorePointCollection.UnmarshalJSON([]byte) error
1. *RestorePointCollectionListResultIterator.Next() error
1. *RestorePointCollectionListResultIterator.NextWithContext(context.Context) error
1. *RestorePointCollectionListResultPage.Next() error
1. *RestorePointCollectionListResultPage.NextWithContext(context.Context) error
1. *RestorePointCollectionUpdate.UnmarshalJSON([]byte) error
1. *RestorePointCollectionsDeleteFuture.UnmarshalJSON([]byte) error
1. *RestorePointsCreateFuture.UnmarshalJSON([]byte) error
1. *RestorePointsDeleteFuture.UnmarshalJSON([]byte) error
1. *SharedGallery.UnmarshalJSON([]byte) error
1. *SharedGalleryImage.UnmarshalJSON([]byte) error
1. *SharedGalleryImageListIterator.Next() error
1. *SharedGalleryImageListIterator.NextWithContext(context.Context) error
1. *SharedGalleryImageListPage.Next() error
1. *SharedGalleryImageListPage.NextWithContext(context.Context) error
1. *SharedGalleryImageVersion.UnmarshalJSON([]byte) error
1. *SharedGalleryImageVersionListIterator.Next() error
1. *SharedGalleryImageVersionListIterator.NextWithContext(context.Context) error
1. *SharedGalleryImageVersionListPage.Next() error
1. *SharedGalleryImageVersionListPage.NextWithContext(context.Context) error
1. *SharedGalleryListIterator.Next() error
1. *SharedGalleryListIterator.NextWithContext(context.Context) error
1. *SharedGalleryListPage.Next() error
1. *SharedGalleryListPage.NextWithContext(context.Context) error
1. *VirtualMachineNetworkInterfaceConfiguration.UnmarshalJSON([]byte) error
1. *VirtualMachineNetworkInterfaceIPConfiguration.UnmarshalJSON([]byte) error
1. *VirtualMachinePublicIPAddressConfiguration.UnmarshalJSON([]byte) error
1. AccessURI.MarshalJSON() ([]byte, error)
1. AvailablePatchSummary.MarshalJSON() ([]byte, error)
1. BootDiagnosticsInstanceView.MarshalJSON() ([]byte, error)
1. CloudServiceRoleProperties.MarshalJSON() ([]byte, error)
1. DataDiskImage.MarshalJSON() ([]byte, error)
1. DiskAccessProperties.MarshalJSON() ([]byte, error)
1. GalleryIdentifier.MarshalJSON() ([]byte, error)
1. GallerySharingProfileClient.Update(context.Context, string, string, SharingUpdate) (GallerySharingProfileUpdateFuture, error)
1. GallerySharingProfileClient.UpdatePreparer(context.Context, string, string, SharingUpdate) (*http.Request, error)
1. GallerySharingProfileClient.UpdateResponder(*http.Response) (SharingUpdate, error)
1. GallerySharingProfileClient.UpdateSender(*http.Request) (GallerySharingProfileUpdateFuture, error)
1. InstanceSku.MarshalJSON() ([]byte, error)
1. InstanceViewStatusesSummary.MarshalJSON() ([]byte, error)
1. LastPatchInstallationSummary.MarshalJSON() ([]byte, error)
1. LogAnalyticsOperationResult.MarshalJSON() ([]byte, error)
1. LogAnalyticsOutput.MarshalJSON() ([]byte, error)
1. NewGallerySharingProfileClient(string) GallerySharingProfileClient
1. NewGallerySharingProfileClientWithBaseURI(string, string) GallerySharingProfileClient
1. NewRestorePointCollectionListResultIterator(RestorePointCollectionListResultPage) RestorePointCollectionListResultIterator
1. NewRestorePointCollectionListResultPage(RestorePointCollectionListResult, func(context.Context, RestorePointCollectionListResult) (RestorePointCollectionListResult, error)) RestorePointCollectionListResultPage
1. NewRestorePointCollectionsClient(string) RestorePointCollectionsClient
1. NewRestorePointCollectionsClientWithBaseURI(string, string) RestorePointCollectionsClient
1. NewRestorePointsClient(string) RestorePointsClient
1. NewRestorePointsClientWithBaseURI(string, string) RestorePointsClient
1. NewSharedGalleriesClient(string) SharedGalleriesClient
1. NewSharedGalleriesClientWithBaseURI(string, string) SharedGalleriesClient
1. NewSharedGalleryImageListIterator(SharedGalleryImageListPage) SharedGalleryImageListIterator
1. NewSharedGalleryImageListPage(SharedGalleryImageList, func(context.Context, SharedGalleryImageList) (SharedGalleryImageList, error)) SharedGalleryImageListPage
1. NewSharedGalleryImageVersionListIterator(SharedGalleryImageVersionListPage) SharedGalleryImageVersionListIterator
1. NewSharedGalleryImageVersionListPage(SharedGalleryImageVersionList, func(context.Context, SharedGalleryImageVersionList) (SharedGalleryImageVersionList, error)) SharedGalleryImageVersionListPage
1. NewSharedGalleryImageVersionsClient(string) SharedGalleryImageVersionsClient
1. NewSharedGalleryImageVersionsClientWithBaseURI(string, string) SharedGalleryImageVersionsClient
1. NewSharedGalleryImagesClient(string) SharedGalleryImagesClient
1. NewSharedGalleryImagesClientWithBaseURI(string, string) SharedGalleryImagesClient
1. NewSharedGalleryListIterator(SharedGalleryListPage) SharedGalleryListIterator
1. NewSharedGalleryListPage(SharedGalleryList, func(context.Context, SharedGalleryList) (SharedGalleryList, error)) SharedGalleryListPage
1. OSFamilyProperties.MarshalJSON() ([]byte, error)
1. OSVersionProperties.MarshalJSON() ([]byte, error)
1. OSVersionPropertiesBase.MarshalJSON() ([]byte, error)
1. OperationListResult.MarshalJSON() ([]byte, error)
1. OperationValueDisplay.MarshalJSON() ([]byte, error)
1. OrchestrationServiceSummary.MarshalJSON() ([]byte, error)
1. PatchInstallationDetail.MarshalJSON() ([]byte, error)
1. PirResource.MarshalJSON() ([]byte, error)
1. PirSharedGalleryResource.MarshalJSON() ([]byte, error)
1. PossibleConsistencyModeTypesValues() []ConsistencyModeTypes
1. PossibleDeleteOptionsValues() []DeleteOptions
1. PossibleDiskDeleteOptionTypesValues() []DiskDeleteOptionTypes
1. PossibleExpandTypesForGetVMScaleSetsValues() []ExpandTypesForGetVMScaleSets
1. PossibleGallerySharingPermissionTypesValues() []GallerySharingPermissionTypes
1. PossibleIPVersionsValues() []IPVersions
1. PossibleLinuxPatchAssessmentModeValues() []LinuxPatchAssessmentMode
1. PossibleNetworkAPIVersionValues() []NetworkAPIVersion
1. PossibleOperatingSystemTypeValues() []OperatingSystemType
1. PossiblePublicIPAddressSkuNameValues() []PublicIPAddressSkuName
1. PossiblePublicIPAddressSkuTierValues() []PublicIPAddressSkuTier
1. PossiblePublicIPAllocationMethodValues() []PublicIPAllocationMethod
1. PossibleRestorePointCollectionExpandOptionsValues() []RestorePointCollectionExpandOptions
1. PossibleSelectPermissionsValues() []SelectPermissions
1. PossibleSharedToValuesValues() []SharedToValues
1. PossibleSharingProfileGroupTypesValues() []SharingProfileGroupTypes
1. PossibleSharingUpdateOperationTypesValues() []SharingUpdateOperationTypes
1. PossibleWindowsPatchAssessmentModeValues() []WindowsPatchAssessmentMode
1. PrivateEndpoint.MarshalJSON() ([]byte, error)
1. ProxyOnlyResource.MarshalJSON() ([]byte, error)
1. ProxyResource.MarshalJSON() ([]byte, error)
1. RecoveryWalkResponse.MarshalJSON() ([]byte, error)
1. RegionalReplicationStatus.MarshalJSON() ([]byte, error)
1. ReplicationStatus.MarshalJSON() ([]byte, error)
1. ResourceSku.MarshalJSON() ([]byte, error)
1. ResourceSkuCapabilities.MarshalJSON() ([]byte, error)
1. ResourceSkuCapacity.MarshalJSON() ([]byte, error)
1. ResourceSkuCosts.MarshalJSON() ([]byte, error)
1. ResourceSkuLocationInfo.MarshalJSON() ([]byte, error)
1. ResourceSkuRestrictionInfo.MarshalJSON() ([]byte, error)
1. ResourceSkuRestrictions.MarshalJSON() ([]byte, error)
1. ResourceSkuZoneDetails.MarshalJSON() ([]byte, error)
1. RestorePoint.MarshalJSON() ([]byte, error)
1. RestorePointCollection.MarshalJSON() ([]byte, error)
1. RestorePointCollectionListResult.IsEmpty() bool
1. RestorePointCollectionListResultIterator.NotDone() bool
1. RestorePointCollectionListResultIterator.Response() RestorePointCollectionListResult
1. RestorePointCollectionListResultIterator.Value() RestorePointCollection
1. RestorePointCollectionListResultPage.NotDone() bool
1. RestorePointCollectionListResultPage.Response() RestorePointCollectionListResult
1. RestorePointCollectionListResultPage.Values() []RestorePointCollection
1. RestorePointCollectionProperties.MarshalJSON() ([]byte, error)
1. RestorePointCollectionSourceProperties.MarshalJSON() ([]byte, error)
1. RestorePointCollectionUpdate.MarshalJSON() ([]byte, error)
1. RestorePointCollectionsClient.CreateOrUpdate(context.Context, string, string, RestorePointCollection) (RestorePointCollection, error)
1. RestorePointCollectionsClient.CreateOrUpdatePreparer(context.Context, string, string, RestorePointCollection) (*http.Request, error)
1. RestorePointCollectionsClient.CreateOrUpdateResponder(*http.Response) (RestorePointCollection, error)
1. RestorePointCollectionsClient.CreateOrUpdateSender(*http.Request) (*http.Response, error)
1. RestorePointCollectionsClient.Delete(context.Context, string, string) (RestorePointCollectionsDeleteFuture, error)
1. RestorePointCollectionsClient.DeletePreparer(context.Context, string, string) (*http.Request, error)
1. RestorePointCollectionsClient.DeleteResponder(*http.Response) (autorest.Response, error)
1. RestorePointCollectionsClient.DeleteSender(*http.Request) (RestorePointCollectionsDeleteFuture, error)
1. RestorePointCollectionsClient.Get(context.Context, string, string, RestorePointCollectionExpandOptions) (RestorePointCollection, error)
1. RestorePointCollectionsClient.GetPreparer(context.Context, string, string, RestorePointCollectionExpandOptions) (*http.Request, error)
1. RestorePointCollectionsClient.GetResponder(*http.Response) (RestorePointCollection, error)
1. RestorePointCollectionsClient.GetSender(*http.Request) (*http.Response, error)
1. RestorePointCollectionsClient.List(context.Context, string) (RestorePointCollectionListResultPage, error)
1. RestorePointCollectionsClient.ListAll(context.Context) (RestorePointCollectionListResultPage, error)
1. RestorePointCollectionsClient.ListAllComplete(context.Context) (RestorePointCollectionListResultIterator, error)
1. RestorePointCollectionsClient.ListAllPreparer(context.Context) (*http.Request, error)
1. RestorePointCollectionsClient.ListAllResponder(*http.Response) (RestorePointCollectionListResult, error)
1. RestorePointCollectionsClient.ListAllSender(*http.Request) (*http.Response, error)
1. RestorePointCollectionsClient.ListComplete(context.Context, string) (RestorePointCollectionListResultIterator, error)
1. RestorePointCollectionsClient.ListPreparer(context.Context, string) (*http.Request, error)
1. RestorePointCollectionsClient.ListResponder(*http.Response) (RestorePointCollectionListResult, error)
1. RestorePointCollectionsClient.ListSender(*http.Request) (*http.Response, error)
1. RestorePointCollectionsClient.Update(context.Context, string, string, RestorePointCollectionUpdate) (RestorePointCollection, error)
1. RestorePointCollectionsClient.UpdatePreparer(context.Context, string, string, RestorePointCollectionUpdate) (*http.Request, error)
1. RestorePointCollectionsClient.UpdateResponder(*http.Response) (RestorePointCollection, error)
1. RestorePointCollectionsClient.UpdateSender(*http.Request) (*http.Response, error)
1. RestorePointsClient.Create(context.Context, string, string, string, RestorePoint) (RestorePointsCreateFuture, error)
1. RestorePointsClient.CreatePreparer(context.Context, string, string, string, RestorePoint) (*http.Request, error)
1. RestorePointsClient.CreateResponder(*http.Response) (RestorePoint, error)
1. RestorePointsClient.CreateSender(*http.Request) (RestorePointsCreateFuture, error)
1. RestorePointsClient.Delete(context.Context, string, string, string) (RestorePointsDeleteFuture, error)
1. RestorePointsClient.DeletePreparer(context.Context, string, string, string) (*http.Request, error)
1. RestorePointsClient.DeleteResponder(*http.Response) (autorest.Response, error)
1. RestorePointsClient.DeleteSender(*http.Request) (RestorePointsDeleteFuture, error)
1. RestorePointsClient.Get(context.Context, string, string, string) (RestorePoint, error)
1. RestorePointsClient.GetPreparer(context.Context, string, string, string) (*http.Request, error)
1. RestorePointsClient.GetResponder(*http.Response) (RestorePoint, error)
1. RestorePointsClient.GetSender(*http.Request) (*http.Response, error)
1. RetrieveBootDiagnosticsDataResult.MarshalJSON() ([]byte, error)
1. RoleInstanceInstanceView.MarshalJSON() ([]byte, error)
1. RoleInstanceNetworkProfile.MarshalJSON() ([]byte, error)
1. RollbackStatusInfo.MarshalJSON() ([]byte, error)
1. RollingUpgradeProgressInfo.MarshalJSON() ([]byte, error)
1. RollingUpgradeRunningStatus.MarshalJSON() ([]byte, error)
1. RollingUpgradeStatusInfoProperties.MarshalJSON() ([]byte, error)
1. ShareInfoElement.MarshalJSON() ([]byte, error)
1. SharedGalleriesClient.Get(context.Context, string, string) (SharedGallery, error)
1. SharedGalleriesClient.GetPreparer(context.Context, string, string) (*http.Request, error)
1. SharedGalleriesClient.GetResponder(*http.Response) (SharedGallery, error)
1. SharedGalleriesClient.GetSender(*http.Request) (*http.Response, error)
1. SharedGalleriesClient.List(context.Context, string, SharedToValues) (SharedGalleryListPage, error)
1. SharedGalleriesClient.ListComplete(context.Context, string, SharedToValues) (SharedGalleryListIterator, error)
1. SharedGalleriesClient.ListPreparer(context.Context, string, SharedToValues) (*http.Request, error)
1. SharedGalleriesClient.ListResponder(*http.Response) (SharedGalleryList, error)
1. SharedGalleriesClient.ListSender(*http.Request) (*http.Response, error)
1. SharedGallery.MarshalJSON() ([]byte, error)
1. SharedGalleryImage.MarshalJSON() ([]byte, error)
1. SharedGalleryImageList.IsEmpty() bool
1. SharedGalleryImageListIterator.NotDone() bool
1. SharedGalleryImageListIterator.Response() SharedGalleryImageList
1. SharedGalleryImageListIterator.Value() SharedGalleryImage
1. SharedGalleryImageListPage.NotDone() bool
1. SharedGalleryImageListPage.Response() SharedGalleryImageList
1. SharedGalleryImageListPage.Values() []SharedGalleryImage
1. SharedGalleryImageVersion.MarshalJSON() ([]byte, error)
1. SharedGalleryImageVersionList.IsEmpty() bool
1. SharedGalleryImageVersionListIterator.NotDone() bool
1. SharedGalleryImageVersionListIterator.Response() SharedGalleryImageVersionList
1. SharedGalleryImageVersionListIterator.Value() SharedGalleryImageVersion
1. SharedGalleryImageVersionListPage.NotDone() bool
1. SharedGalleryImageVersionListPage.Response() SharedGalleryImageVersionList
1. SharedGalleryImageVersionListPage.Values() []SharedGalleryImageVersion
1. SharedGalleryImageVersionsClient.Get(context.Context, string, string, string, string) (SharedGalleryImageVersion, error)
1. SharedGalleryImageVersionsClient.GetPreparer(context.Context, string, string, string, string) (*http.Request, error)
1. SharedGalleryImageVersionsClient.GetResponder(*http.Response) (SharedGalleryImageVersion, error)
1. SharedGalleryImageVersionsClient.GetSender(*http.Request) (*http.Response, error)
1. SharedGalleryImageVersionsClient.List(context.Context, string, string, string, SharedToValues) (SharedGalleryImageVersionListPage, error)
1. SharedGalleryImageVersionsClient.ListComplete(context.Context, string, string, string, SharedToValues) (SharedGalleryImageVersionListIterator, error)
1. SharedGalleryImageVersionsClient.ListPreparer(context.Context, string, string, string, SharedToValues) (*http.Request, error)
1. SharedGalleryImageVersionsClient.ListResponder(*http.Response) (SharedGalleryImageVersionList, error)
1. SharedGalleryImageVersionsClient.ListSender(*http.Request) (*http.Response, error)
1. SharedGalleryImagesClient.Get(context.Context, string, string, string) (SharedGalleryImage, error)
1. SharedGalleryImagesClient.GetPreparer(context.Context, string, string, string) (*http.Request, error)
1. SharedGalleryImagesClient.GetResponder(*http.Response) (SharedGalleryImage, error)
1. SharedGalleryImagesClient.GetSender(*http.Request) (*http.Response, error)
1. SharedGalleryImagesClient.List(context.Context, string, string, SharedToValues) (SharedGalleryImageListPage, error)
1. SharedGalleryImagesClient.ListComplete(context.Context, string, string, SharedToValues) (SharedGalleryImageListIterator, error)
1. SharedGalleryImagesClient.ListPreparer(context.Context, string, string, SharedToValues) (*http.Request, error)
1. SharedGalleryImagesClient.ListResponder(*http.Response) (SharedGalleryImageList, error)
1. SharedGalleryImagesClient.ListSender(*http.Request) (*http.Response, error)
1. SharedGalleryList.IsEmpty() bool
1. SharedGalleryListIterator.NotDone() bool
1. SharedGalleryListIterator.Response() SharedGalleryList
1. SharedGalleryListIterator.Value() SharedGallery
1. SharedGalleryListPage.NotDone() bool
1. SharedGalleryListPage.Response() SharedGalleryList
1. SharedGalleryListPage.Values() []SharedGallery
1. SharingProfile.MarshalJSON() ([]byte, error)
1. StatusCodeCount.MarshalJSON() ([]byte, error)
1. SubResourceReadOnly.MarshalJSON() ([]byte, error)
1. UpdateDomain.MarshalJSON() ([]byte, error)
1. UpgradeOperationHistoricalStatusInfo.MarshalJSON() ([]byte, error)
1. UpgradeOperationHistoricalStatusInfoProperties.MarshalJSON() ([]byte, error)
1. UpgradeOperationHistoryStatus.MarshalJSON() ([]byte, error)
1. VirtualMachineAssessPatchesResult.MarshalJSON() ([]byte, error)
1. VirtualMachineHealthStatus.MarshalJSON() ([]byte, error)
1. VirtualMachineIdentityUserAssignedIdentitiesValue.MarshalJSON() ([]byte, error)
1. VirtualMachineInstallPatchesResult.MarshalJSON() ([]byte, error)
1. VirtualMachineNetworkInterfaceConfiguration.MarshalJSON() ([]byte, error)
1. VirtualMachineNetworkInterfaceIPConfiguration.MarshalJSON() ([]byte, error)
1. VirtualMachinePublicIPAddressConfiguration.MarshalJSON() ([]byte, error)
1. VirtualMachineScaleSetIdentityUserAssignedIdentitiesValue.MarshalJSON() ([]byte, error)
1. VirtualMachineScaleSetInstanceViewStatusesSummary.MarshalJSON() ([]byte, error)
1. VirtualMachineScaleSetSku.MarshalJSON() ([]byte, error)
1. VirtualMachineScaleSetSkuCapacity.MarshalJSON() ([]byte, error)
1. VirtualMachineScaleSetVMExtensionsSummary.MarshalJSON() ([]byte, error)
1. VirtualMachineSoftwarePatchProperties.MarshalJSON() ([]byte, error)
1. VirtualMachineStatusCodeCount.MarshalJSON() ([]byte, error)

### Struct Changes

#### New Structs

1. GalleryImageFeature
1. GallerySharingProfileClient
1. GallerySharingProfileUpdateFuture
1. PirResource
1. PirSharedGalleryResource
1. ProxyResource
1. PublicIPAddressSku
1. RestorePoint
1. RestorePointCollection
1. RestorePointCollectionListResult
1. RestorePointCollectionListResultIterator
1. RestorePointCollectionListResultPage
1. RestorePointCollectionProperties
1. RestorePointCollectionSourceProperties
1. RestorePointCollectionUpdate
1. RestorePointCollectionsClient
1. RestorePointCollectionsDeleteFuture
1. RestorePointProvisioningDetails
1. RestorePointSourceMetadata
1. RestorePointSourceVMDataDisk
1. RestorePointSourceVMOSDisk
1. RestorePointSourceVMStorageProfile
1. RestorePointsClient
1. RestorePointsCreateFuture
1. RestorePointsDeleteFuture
1. SharedGalleriesClient
1. SharedGallery
1. SharedGalleryIdentifier
1. SharedGalleryImage
1. SharedGalleryImageList
1. SharedGalleryImageListIterator
1. SharedGalleryImageListPage
1. SharedGalleryImageProperties
1. SharedGalleryImageVersion
1. SharedGalleryImageVersionList
1. SharedGalleryImageVersionListIterator
1. SharedGalleryImageVersionListPage
1. SharedGalleryImageVersionProperties
1. SharedGalleryImageVersionsClient
1. SharedGalleryImagesClient
1. SharedGalleryList
1. SharedGalleryListIterator
1. SharedGalleryListPage
1. SharingProfile
1. SharingProfileGroup
1. SharingUpdate
1. VirtualMachineIPTag
1. VirtualMachineNetworkInterfaceConfiguration
1. VirtualMachineNetworkInterfaceConfigurationProperties
1. VirtualMachineNetworkInterfaceDNSSettingsConfiguration
1. VirtualMachineNetworkInterfaceIPConfiguration
1. VirtualMachineNetworkInterfaceIPConfigurationProperties
1. VirtualMachinePublicIPAddressConfiguration
1. VirtualMachinePublicIPAddressConfigurationProperties
1. VirtualMachinePublicIPAddressDNSSettingsConfiguration

#### New Struct Fields

1. DataDisk.DeleteOption
1. GalleryArtifactVersionSource.URI
1. GalleryImageProperties.Features
1. GalleryProperties.SharingProfile
1. LinuxPatchSettings.AssessmentMode
1. NetworkInterfaceReferenceProperties.DeleteOption
1. NetworkProfile.NetworkAPIVersion
1. NetworkProfile.NetworkInterfaceConfigurations
1. OSDisk.DeleteOption
1. PatchSettings.AssessmentMode
1. VirtualMachineProperties.ScheduledEventsProfile
1. VirtualMachineProperties.UserData
1. VirtualMachineScaleSetNetworkConfigurationProperties.DeleteOption
1. VirtualMachineScaleSetNetworkProfile.NetworkAPIVersion
1. VirtualMachineScaleSetPublicIPAddressConfiguration.Sku
1. VirtualMachineScaleSetPublicIPAddressConfigurationProperties.DeleteOption
1. VirtualMachineScaleSetUpdateNetworkConfigurationProperties.DeleteOption
1. VirtualMachineScaleSetUpdateNetworkProfile.NetworkAPIVersion
1. VirtualMachineScaleSetUpdatePublicIPAddressConfigurationProperties.DeleteOption
1. VirtualMachineScaleSetUpdateVMProfile.UserData
1. VirtualMachineScaleSetVMProfile.UserData
1. VirtualMachineScaleSetVMProperties.UserData
