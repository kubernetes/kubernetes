/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package types

import "reflect"

type ActionParameter string

const (
	ActionParameterTargetName        = ActionParameter("targetName")
	ActionParameterAlarmName         = ActionParameter("alarmName")
	ActionParameterOldStatus         = ActionParameter("oldStatus")
	ActionParameterNewStatus         = ActionParameter("newStatus")
	ActionParameterTriggeringSummary = ActionParameter("triggeringSummary")
	ActionParameterDeclaringSummary  = ActionParameter("declaringSummary")
	ActionParameterEventDescription  = ActionParameter("eventDescription")
	ActionParameterTarget            = ActionParameter("target")
	ActionParameterAlarm             = ActionParameter("alarm")
)

func init() {
	t["ActionParameter"] = reflect.TypeOf((*ActionParameter)(nil)).Elem()
}

type ActionType string

const (
	ActionTypeMigrationV1         = ActionType("MigrationV1")
	ActionTypeVmPowerV1           = ActionType("VmPowerV1")
	ActionTypeHostPowerV1         = ActionType("HostPowerV1")
	ActionTypeHostMaintenanceV1   = ActionType("HostMaintenanceV1")
	ActionTypeStorageMigrationV1  = ActionType("StorageMigrationV1")
	ActionTypeStoragePlacementV1  = ActionType("StoragePlacementV1")
	ActionTypePlacementV1         = ActionType("PlacementV1")
	ActionTypeHostInfraUpdateHaV1 = ActionType("HostInfraUpdateHaV1")
)

func init() {
	t["ActionType"] = reflect.TypeOf((*ActionType)(nil)).Elem()
}

type AffinityType string

const (
	AffinityTypeMemory = AffinityType("memory")
	AffinityTypeCpu    = AffinityType("cpu")
)

func init() {
	t["AffinityType"] = reflect.TypeOf((*AffinityType)(nil)).Elem()
}

type AgentInstallFailedReason string

const (
	AgentInstallFailedReasonNotEnoughSpaceOnDevice      = AgentInstallFailedReason("NotEnoughSpaceOnDevice")
	AgentInstallFailedReasonPrepareToUpgradeFailed      = AgentInstallFailedReason("PrepareToUpgradeFailed")
	AgentInstallFailedReasonAgentNotRunning             = AgentInstallFailedReason("AgentNotRunning")
	AgentInstallFailedReasonAgentNotReachable           = AgentInstallFailedReason("AgentNotReachable")
	AgentInstallFailedReasonInstallTimedout             = AgentInstallFailedReason("InstallTimedout")
	AgentInstallFailedReasonSignatureVerificationFailed = AgentInstallFailedReason("SignatureVerificationFailed")
	AgentInstallFailedReasonAgentUploadFailed           = AgentInstallFailedReason("AgentUploadFailed")
	AgentInstallFailedReasonAgentUploadTimedout         = AgentInstallFailedReason("AgentUploadTimedout")
	AgentInstallFailedReasonUnknownInstallerError       = AgentInstallFailedReason("UnknownInstallerError")
)

func init() {
	t["AgentInstallFailedReason"] = reflect.TypeOf((*AgentInstallFailedReason)(nil)).Elem()
}

type ArrayUpdateOperation string

const (
	ArrayUpdateOperationAdd    = ArrayUpdateOperation("add")
	ArrayUpdateOperationRemove = ArrayUpdateOperation("remove")
	ArrayUpdateOperationEdit   = ArrayUpdateOperation("edit")
)

func init() {
	t["ArrayUpdateOperation"] = reflect.TypeOf((*ArrayUpdateOperation)(nil)).Elem()
}

type AutoStartAction string

const (
	AutoStartActionNone          = AutoStartAction("none")
	AutoStartActionSystemDefault = AutoStartAction("systemDefault")
	AutoStartActionPowerOn       = AutoStartAction("powerOn")
	AutoStartActionPowerOff      = AutoStartAction("powerOff")
	AutoStartActionGuestShutdown = AutoStartAction("guestShutdown")
	AutoStartActionSuspend       = AutoStartAction("suspend")
)

func init() {
	t["AutoStartAction"] = reflect.TypeOf((*AutoStartAction)(nil)).Elem()
}

type AutoStartWaitHeartbeatSetting string

const (
	AutoStartWaitHeartbeatSettingYes           = AutoStartWaitHeartbeatSetting("yes")
	AutoStartWaitHeartbeatSettingNo            = AutoStartWaitHeartbeatSetting("no")
	AutoStartWaitHeartbeatSettingSystemDefault = AutoStartWaitHeartbeatSetting("systemDefault")
)

func init() {
	t["AutoStartWaitHeartbeatSetting"] = reflect.TypeOf((*AutoStartWaitHeartbeatSetting)(nil)).Elem()
}

type BaseConfigInfoDiskFileBackingInfoProvisioningType string

const (
	BaseConfigInfoDiskFileBackingInfoProvisioningTypeThin             = BaseConfigInfoDiskFileBackingInfoProvisioningType("thin")
	BaseConfigInfoDiskFileBackingInfoProvisioningTypeEagerZeroedThick = BaseConfigInfoDiskFileBackingInfoProvisioningType("eagerZeroedThick")
	BaseConfigInfoDiskFileBackingInfoProvisioningTypeLazyZeroedThick  = BaseConfigInfoDiskFileBackingInfoProvisioningType("lazyZeroedThick")
)

func init() {
	t["BaseConfigInfoDiskFileBackingInfoProvisioningType"] = reflect.TypeOf((*BaseConfigInfoDiskFileBackingInfoProvisioningType)(nil)).Elem()
}

type BatchResultResult string

const (
	BatchResultResultSuccess = BatchResultResult("success")
	BatchResultResultFail    = BatchResultResult("fail")
)

func init() {
	t["BatchResultResult"] = reflect.TypeOf((*BatchResultResult)(nil)).Elem()
}

type CannotEnableVmcpForClusterReason string

const (
	CannotEnableVmcpForClusterReasonAPDTimeoutDisabled      = CannotEnableVmcpForClusterReason("APDTimeoutDisabled")
	CannotEnableVmcpForClusterReasonIncompatibleHostVersion = CannotEnableVmcpForClusterReason("IncompatibleHostVersion")
)

func init() {
	t["CannotEnableVmcpForClusterReason"] = reflect.TypeOf((*CannotEnableVmcpForClusterReason)(nil)).Elem()
}

type CannotMoveFaultToleranceVmMoveType string

const (
	CannotMoveFaultToleranceVmMoveTypeResourcePool = CannotMoveFaultToleranceVmMoveType("resourcePool")
	CannotMoveFaultToleranceVmMoveTypeCluster      = CannotMoveFaultToleranceVmMoveType("cluster")
)

func init() {
	t["CannotMoveFaultToleranceVmMoveType"] = reflect.TypeOf((*CannotMoveFaultToleranceVmMoveType)(nil)).Elem()
}

type CannotPowerOffVmInClusterOperation string

const (
	CannotPowerOffVmInClusterOperationSuspend       = CannotPowerOffVmInClusterOperation("suspend")
	CannotPowerOffVmInClusterOperationPowerOff      = CannotPowerOffVmInClusterOperation("powerOff")
	CannotPowerOffVmInClusterOperationGuestShutdown = CannotPowerOffVmInClusterOperation("guestShutdown")
	CannotPowerOffVmInClusterOperationGuestSuspend  = CannotPowerOffVmInClusterOperation("guestSuspend")
)

func init() {
	t["CannotPowerOffVmInClusterOperation"] = reflect.TypeOf((*CannotPowerOffVmInClusterOperation)(nil)).Elem()
}

type CannotUseNetworkReason string

const (
	CannotUseNetworkReasonNetworkReservationNotSupported  = CannotUseNetworkReason("NetworkReservationNotSupported")
	CannotUseNetworkReasonMismatchedNetworkPolicies       = CannotUseNetworkReason("MismatchedNetworkPolicies")
	CannotUseNetworkReasonMismatchedDvsVersionOrVendor    = CannotUseNetworkReason("MismatchedDvsVersionOrVendor")
	CannotUseNetworkReasonVMotionToUnsupportedNetworkType = CannotUseNetworkReason("VMotionToUnsupportedNetworkType")
)

func init() {
	t["CannotUseNetworkReason"] = reflect.TypeOf((*CannotUseNetworkReason)(nil)).Elem()
}

type CheckTestType string

const (
	CheckTestTypeSourceTests       = CheckTestType("sourceTests")
	CheckTestTypeHostTests         = CheckTestType("hostTests")
	CheckTestTypeResourcePoolTests = CheckTestType("resourcePoolTests")
	CheckTestTypeDatastoreTests    = CheckTestType("datastoreTests")
	CheckTestTypeNetworkTests      = CheckTestType("networkTests")
)

func init() {
	t["CheckTestType"] = reflect.TypeOf((*CheckTestType)(nil)).Elem()
}

type ClusterDasAamNodeStateDasState string

const (
	ClusterDasAamNodeStateDasStateUninitialized = ClusterDasAamNodeStateDasState("uninitialized")
	ClusterDasAamNodeStateDasStateInitialized   = ClusterDasAamNodeStateDasState("initialized")
	ClusterDasAamNodeStateDasStateConfiguring   = ClusterDasAamNodeStateDasState("configuring")
	ClusterDasAamNodeStateDasStateUnconfiguring = ClusterDasAamNodeStateDasState("unconfiguring")
	ClusterDasAamNodeStateDasStateRunning       = ClusterDasAamNodeStateDasState("running")
	ClusterDasAamNodeStateDasStateError         = ClusterDasAamNodeStateDasState("error")
	ClusterDasAamNodeStateDasStateAgentShutdown = ClusterDasAamNodeStateDasState("agentShutdown")
	ClusterDasAamNodeStateDasStateNodeFailed    = ClusterDasAamNodeStateDasState("nodeFailed")
)

func init() {
	t["ClusterDasAamNodeStateDasState"] = reflect.TypeOf((*ClusterDasAamNodeStateDasState)(nil)).Elem()
}

type ClusterDasConfigInfoHBDatastoreCandidate string

const (
	ClusterDasConfigInfoHBDatastoreCandidateUserSelectedDs                  = ClusterDasConfigInfoHBDatastoreCandidate("userSelectedDs")
	ClusterDasConfigInfoHBDatastoreCandidateAllFeasibleDs                   = ClusterDasConfigInfoHBDatastoreCandidate("allFeasibleDs")
	ClusterDasConfigInfoHBDatastoreCandidateAllFeasibleDsWithUserPreference = ClusterDasConfigInfoHBDatastoreCandidate("allFeasibleDsWithUserPreference")
)

func init() {
	t["ClusterDasConfigInfoHBDatastoreCandidate"] = reflect.TypeOf((*ClusterDasConfigInfoHBDatastoreCandidate)(nil)).Elem()
}

type ClusterDasConfigInfoServiceState string

const (
	ClusterDasConfigInfoServiceStateDisabled = ClusterDasConfigInfoServiceState("disabled")
	ClusterDasConfigInfoServiceStateEnabled  = ClusterDasConfigInfoServiceState("enabled")
)

func init() {
	t["ClusterDasConfigInfoServiceState"] = reflect.TypeOf((*ClusterDasConfigInfoServiceState)(nil)).Elem()
}

type ClusterDasConfigInfoVmMonitoringState string

const (
	ClusterDasConfigInfoVmMonitoringStateVmMonitoringDisabled = ClusterDasConfigInfoVmMonitoringState("vmMonitoringDisabled")
	ClusterDasConfigInfoVmMonitoringStateVmMonitoringOnly     = ClusterDasConfigInfoVmMonitoringState("vmMonitoringOnly")
	ClusterDasConfigInfoVmMonitoringStateVmAndAppMonitoring   = ClusterDasConfigInfoVmMonitoringState("vmAndAppMonitoring")
)

func init() {
	t["ClusterDasConfigInfoVmMonitoringState"] = reflect.TypeOf((*ClusterDasConfigInfoVmMonitoringState)(nil)).Elem()
}

type ClusterDasFdmAvailabilityState string

const (
	ClusterDasFdmAvailabilityStateUninitialized                = ClusterDasFdmAvailabilityState("uninitialized")
	ClusterDasFdmAvailabilityStateElection                     = ClusterDasFdmAvailabilityState("election")
	ClusterDasFdmAvailabilityStateMaster                       = ClusterDasFdmAvailabilityState("master")
	ClusterDasFdmAvailabilityStateConnectedToMaster            = ClusterDasFdmAvailabilityState("connectedToMaster")
	ClusterDasFdmAvailabilityStateNetworkPartitionedFromMaster = ClusterDasFdmAvailabilityState("networkPartitionedFromMaster")
	ClusterDasFdmAvailabilityStateNetworkIsolated              = ClusterDasFdmAvailabilityState("networkIsolated")
	ClusterDasFdmAvailabilityStateHostDown                     = ClusterDasFdmAvailabilityState("hostDown")
	ClusterDasFdmAvailabilityStateInitializationError          = ClusterDasFdmAvailabilityState("initializationError")
	ClusterDasFdmAvailabilityStateUninitializationError        = ClusterDasFdmAvailabilityState("uninitializationError")
	ClusterDasFdmAvailabilityStateFdmUnreachable               = ClusterDasFdmAvailabilityState("fdmUnreachable")
)

func init() {
	t["ClusterDasFdmAvailabilityState"] = reflect.TypeOf((*ClusterDasFdmAvailabilityState)(nil)).Elem()
}

type ClusterDasVmSettingsIsolationResponse string

const (
	ClusterDasVmSettingsIsolationResponseNone                     = ClusterDasVmSettingsIsolationResponse("none")
	ClusterDasVmSettingsIsolationResponsePowerOff                 = ClusterDasVmSettingsIsolationResponse("powerOff")
	ClusterDasVmSettingsIsolationResponseShutdown                 = ClusterDasVmSettingsIsolationResponse("shutdown")
	ClusterDasVmSettingsIsolationResponseClusterIsolationResponse = ClusterDasVmSettingsIsolationResponse("clusterIsolationResponse")
)

func init() {
	t["ClusterDasVmSettingsIsolationResponse"] = reflect.TypeOf((*ClusterDasVmSettingsIsolationResponse)(nil)).Elem()
}

type ClusterDasVmSettingsRestartPriority string

const (
	ClusterDasVmSettingsRestartPriorityDisabled               = ClusterDasVmSettingsRestartPriority("disabled")
	ClusterDasVmSettingsRestartPriorityLowest                 = ClusterDasVmSettingsRestartPriority("lowest")
	ClusterDasVmSettingsRestartPriorityLow                    = ClusterDasVmSettingsRestartPriority("low")
	ClusterDasVmSettingsRestartPriorityMedium                 = ClusterDasVmSettingsRestartPriority("medium")
	ClusterDasVmSettingsRestartPriorityHigh                   = ClusterDasVmSettingsRestartPriority("high")
	ClusterDasVmSettingsRestartPriorityHighest                = ClusterDasVmSettingsRestartPriority("highest")
	ClusterDasVmSettingsRestartPriorityClusterRestartPriority = ClusterDasVmSettingsRestartPriority("clusterRestartPriority")
)

func init() {
	t["ClusterDasVmSettingsRestartPriority"] = reflect.TypeOf((*ClusterDasVmSettingsRestartPriority)(nil)).Elem()
}

type ClusterHostInfraUpdateHaModeActionOperationType string

const (
	ClusterHostInfraUpdateHaModeActionOperationTypeEnterQuarantine  = ClusterHostInfraUpdateHaModeActionOperationType("enterQuarantine")
	ClusterHostInfraUpdateHaModeActionOperationTypeExitQuarantine   = ClusterHostInfraUpdateHaModeActionOperationType("exitQuarantine")
	ClusterHostInfraUpdateHaModeActionOperationTypeEnterMaintenance = ClusterHostInfraUpdateHaModeActionOperationType("enterMaintenance")
)

func init() {
	t["ClusterHostInfraUpdateHaModeActionOperationType"] = reflect.TypeOf((*ClusterHostInfraUpdateHaModeActionOperationType)(nil)).Elem()
}

type ClusterInfraUpdateHaConfigInfoBehaviorType string

const (
	ClusterInfraUpdateHaConfigInfoBehaviorTypeManual    = ClusterInfraUpdateHaConfigInfoBehaviorType("Manual")
	ClusterInfraUpdateHaConfigInfoBehaviorTypeAutomated = ClusterInfraUpdateHaConfigInfoBehaviorType("Automated")
)

func init() {
	t["ClusterInfraUpdateHaConfigInfoBehaviorType"] = reflect.TypeOf((*ClusterInfraUpdateHaConfigInfoBehaviorType)(nil)).Elem()
}

type ClusterInfraUpdateHaConfigInfoRemediationType string

const (
	ClusterInfraUpdateHaConfigInfoRemediationTypeQuarantineMode  = ClusterInfraUpdateHaConfigInfoRemediationType("QuarantineMode")
	ClusterInfraUpdateHaConfigInfoRemediationTypeMaintenanceMode = ClusterInfraUpdateHaConfigInfoRemediationType("MaintenanceMode")
)

func init() {
	t["ClusterInfraUpdateHaConfigInfoRemediationType"] = reflect.TypeOf((*ClusterInfraUpdateHaConfigInfoRemediationType)(nil)).Elem()
}

type ClusterPowerOnVmOption string

const (
	ClusterPowerOnVmOptionOverrideAutomationLevel = ClusterPowerOnVmOption("OverrideAutomationLevel")
	ClusterPowerOnVmOptionReserveResources        = ClusterPowerOnVmOption("ReserveResources")
)

func init() {
	t["ClusterPowerOnVmOption"] = reflect.TypeOf((*ClusterPowerOnVmOption)(nil)).Elem()
}

type ClusterProfileServiceType string

const (
	ClusterProfileServiceTypeDRS = ClusterProfileServiceType("DRS")
	ClusterProfileServiceTypeHA  = ClusterProfileServiceType("HA")
	ClusterProfileServiceTypeDPM = ClusterProfileServiceType("DPM")
	ClusterProfileServiceTypeFT  = ClusterProfileServiceType("FT")
)

func init() {
	t["ClusterProfileServiceType"] = reflect.TypeOf((*ClusterProfileServiceType)(nil)).Elem()
}

type ClusterVmComponentProtectionSettingsStorageVmReaction string

const (
	ClusterVmComponentProtectionSettingsStorageVmReactionDisabled            = ClusterVmComponentProtectionSettingsStorageVmReaction("disabled")
	ClusterVmComponentProtectionSettingsStorageVmReactionWarning             = ClusterVmComponentProtectionSettingsStorageVmReaction("warning")
	ClusterVmComponentProtectionSettingsStorageVmReactionRestartConservative = ClusterVmComponentProtectionSettingsStorageVmReaction("restartConservative")
	ClusterVmComponentProtectionSettingsStorageVmReactionRestartAggressive   = ClusterVmComponentProtectionSettingsStorageVmReaction("restartAggressive")
	ClusterVmComponentProtectionSettingsStorageVmReactionClusterDefault      = ClusterVmComponentProtectionSettingsStorageVmReaction("clusterDefault")
)

func init() {
	t["ClusterVmComponentProtectionSettingsStorageVmReaction"] = reflect.TypeOf((*ClusterVmComponentProtectionSettingsStorageVmReaction)(nil)).Elem()
}

type ClusterVmComponentProtectionSettingsVmReactionOnAPDCleared string

const (
	ClusterVmComponentProtectionSettingsVmReactionOnAPDClearedNone              = ClusterVmComponentProtectionSettingsVmReactionOnAPDCleared("none")
	ClusterVmComponentProtectionSettingsVmReactionOnAPDClearedReset             = ClusterVmComponentProtectionSettingsVmReactionOnAPDCleared("reset")
	ClusterVmComponentProtectionSettingsVmReactionOnAPDClearedUseClusterDefault = ClusterVmComponentProtectionSettingsVmReactionOnAPDCleared("useClusterDefault")
)

func init() {
	t["ClusterVmComponentProtectionSettingsVmReactionOnAPDCleared"] = reflect.TypeOf((*ClusterVmComponentProtectionSettingsVmReactionOnAPDCleared)(nil)).Elem()
}

type ClusterVmReadinessReadyCondition string

const (
	ClusterVmReadinessReadyConditionNone               = ClusterVmReadinessReadyCondition("none")
	ClusterVmReadinessReadyConditionPoweredOn          = ClusterVmReadinessReadyCondition("poweredOn")
	ClusterVmReadinessReadyConditionGuestHbStatusGreen = ClusterVmReadinessReadyCondition("guestHbStatusGreen")
	ClusterVmReadinessReadyConditionAppHbStatusGreen   = ClusterVmReadinessReadyCondition("appHbStatusGreen")
	ClusterVmReadinessReadyConditionUseClusterDefault  = ClusterVmReadinessReadyCondition("useClusterDefault")
)

func init() {
	t["ClusterVmReadinessReadyCondition"] = reflect.TypeOf((*ClusterVmReadinessReadyCondition)(nil)).Elem()
}

type ComplianceResultStatus string

const (
	ComplianceResultStatusCompliant    = ComplianceResultStatus("compliant")
	ComplianceResultStatusNonCompliant = ComplianceResultStatus("nonCompliant")
	ComplianceResultStatusUnknown      = ComplianceResultStatus("unknown")
)

func init() {
	t["ComplianceResultStatus"] = reflect.TypeOf((*ComplianceResultStatus)(nil)).Elem()
}

type ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseState string

const (
	ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseStateLicensed   = ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseState("licensed")
	ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseStateUnlicensed = ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseState("unlicensed")
	ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseStateUnknown    = ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseState("unknown")
)

func init() {
	t["ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseState"] = reflect.TypeOf((*ComputeResourceHostSPBMLicenseInfoHostSPBMLicenseState)(nil)).Elem()
}

type ConfigSpecOperation string

const (
	ConfigSpecOperationAdd    = ConfigSpecOperation("add")
	ConfigSpecOperationEdit   = ConfigSpecOperation("edit")
	ConfigSpecOperationRemove = ConfigSpecOperation("remove")
)

func init() {
	t["ConfigSpecOperation"] = reflect.TypeOf((*ConfigSpecOperation)(nil)).Elem()
}

type CustomizationLicenseDataMode string

const (
	CustomizationLicenseDataModePerServer = CustomizationLicenseDataMode("perServer")
	CustomizationLicenseDataModePerSeat   = CustomizationLicenseDataMode("perSeat")
)

func init() {
	t["CustomizationLicenseDataMode"] = reflect.TypeOf((*CustomizationLicenseDataMode)(nil)).Elem()
}

type CustomizationNetBIOSMode string

const (
	CustomizationNetBIOSModeEnableNetBIOSViaDhcp = CustomizationNetBIOSMode("enableNetBIOSViaDhcp")
	CustomizationNetBIOSModeEnableNetBIOS        = CustomizationNetBIOSMode("enableNetBIOS")
	CustomizationNetBIOSModeDisableNetBIOS       = CustomizationNetBIOSMode("disableNetBIOS")
)

func init() {
	t["CustomizationNetBIOSMode"] = reflect.TypeOf((*CustomizationNetBIOSMode)(nil)).Elem()
}

type CustomizationSysprepRebootOption string

const (
	CustomizationSysprepRebootOptionReboot   = CustomizationSysprepRebootOption("reboot")
	CustomizationSysprepRebootOptionNoreboot = CustomizationSysprepRebootOption("noreboot")
	CustomizationSysprepRebootOptionShutdown = CustomizationSysprepRebootOption("shutdown")
)

func init() {
	t["CustomizationSysprepRebootOption"] = reflect.TypeOf((*CustomizationSysprepRebootOption)(nil)).Elem()
}

type DVPortStatusVmDirectPathGen2InactiveReasonNetwork string

const (
	DVPortStatusVmDirectPathGen2InactiveReasonNetworkPortNptIncompatibleDvs             = DVPortStatusVmDirectPathGen2InactiveReasonNetwork("portNptIncompatibleDvs")
	DVPortStatusVmDirectPathGen2InactiveReasonNetworkPortNptNoCompatibleNics            = DVPortStatusVmDirectPathGen2InactiveReasonNetwork("portNptNoCompatibleNics")
	DVPortStatusVmDirectPathGen2InactiveReasonNetworkPortNptNoVirtualFunctionsAvailable = DVPortStatusVmDirectPathGen2InactiveReasonNetwork("portNptNoVirtualFunctionsAvailable")
	DVPortStatusVmDirectPathGen2InactiveReasonNetworkPortNptDisabledForPort             = DVPortStatusVmDirectPathGen2InactiveReasonNetwork("portNptDisabledForPort")
)

func init() {
	t["DVPortStatusVmDirectPathGen2InactiveReasonNetwork"] = reflect.TypeOf((*DVPortStatusVmDirectPathGen2InactiveReasonNetwork)(nil)).Elem()
}

type DVPortStatusVmDirectPathGen2InactiveReasonOther string

const (
	DVPortStatusVmDirectPathGen2InactiveReasonOtherPortNptIncompatibleHost      = DVPortStatusVmDirectPathGen2InactiveReasonOther("portNptIncompatibleHost")
	DVPortStatusVmDirectPathGen2InactiveReasonOtherPortNptIncompatibleConnectee = DVPortStatusVmDirectPathGen2InactiveReasonOther("portNptIncompatibleConnectee")
)

func init() {
	t["DVPortStatusVmDirectPathGen2InactiveReasonOther"] = reflect.TypeOf((*DVPortStatusVmDirectPathGen2InactiveReasonOther)(nil)).Elem()
}

type DasConfigFaultDasConfigFaultReason string

const (
	DasConfigFaultDasConfigFaultReasonHostNetworkMisconfiguration = DasConfigFaultDasConfigFaultReason("HostNetworkMisconfiguration")
	DasConfigFaultDasConfigFaultReasonHostMisconfiguration        = DasConfigFaultDasConfigFaultReason("HostMisconfiguration")
	DasConfigFaultDasConfigFaultReasonInsufficientPrivileges      = DasConfigFaultDasConfigFaultReason("InsufficientPrivileges")
	DasConfigFaultDasConfigFaultReasonNoPrimaryAgentAvailable     = DasConfigFaultDasConfigFaultReason("NoPrimaryAgentAvailable")
	DasConfigFaultDasConfigFaultReasonOther                       = DasConfigFaultDasConfigFaultReason("Other")
	DasConfigFaultDasConfigFaultReasonNoDatastoresConfigured      = DasConfigFaultDasConfigFaultReason("NoDatastoresConfigured")
	DasConfigFaultDasConfigFaultReasonCreateConfigVvolFailed      = DasConfigFaultDasConfigFaultReason("CreateConfigVvolFailed")
	DasConfigFaultDasConfigFaultReasonVSanNotSupportedOnHost      = DasConfigFaultDasConfigFaultReason("VSanNotSupportedOnHost")
	DasConfigFaultDasConfigFaultReasonDasNetworkMisconfiguration  = DasConfigFaultDasConfigFaultReason("DasNetworkMisconfiguration")
)

func init() {
	t["DasConfigFaultDasConfigFaultReason"] = reflect.TypeOf((*DasConfigFaultDasConfigFaultReason)(nil)).Elem()
}

type DasVmPriority string

const (
	DasVmPriorityDisabled = DasVmPriority("disabled")
	DasVmPriorityLow      = DasVmPriority("low")
	DasVmPriorityMedium   = DasVmPriority("medium")
	DasVmPriorityHigh     = DasVmPriority("high")
)

func init() {
	t["DasVmPriority"] = reflect.TypeOf((*DasVmPriority)(nil)).Elem()
}

type DatastoreAccessible string

const (
	DatastoreAccessibleTrue  = DatastoreAccessible("True")
	DatastoreAccessibleFalse = DatastoreAccessible("False")
)

func init() {
	t["DatastoreAccessible"] = reflect.TypeOf((*DatastoreAccessible)(nil)).Elem()
}

type DatastoreSummaryMaintenanceModeState string

const (
	DatastoreSummaryMaintenanceModeStateNormal              = DatastoreSummaryMaintenanceModeState("normal")
	DatastoreSummaryMaintenanceModeStateEnteringMaintenance = DatastoreSummaryMaintenanceModeState("enteringMaintenance")
	DatastoreSummaryMaintenanceModeStateInMaintenance       = DatastoreSummaryMaintenanceModeState("inMaintenance")
)

func init() {
	t["DatastoreSummaryMaintenanceModeState"] = reflect.TypeOf((*DatastoreSummaryMaintenanceModeState)(nil)).Elem()
}

type DayOfWeek string

const (
	DayOfWeekSunday    = DayOfWeek("sunday")
	DayOfWeekMonday    = DayOfWeek("monday")
	DayOfWeekTuesday   = DayOfWeek("tuesday")
	DayOfWeekWednesday = DayOfWeek("wednesday")
	DayOfWeekThursday  = DayOfWeek("thursday")
	DayOfWeekFriday    = DayOfWeek("friday")
	DayOfWeekSaturday  = DayOfWeek("saturday")
)

func init() {
	t["DayOfWeek"] = reflect.TypeOf((*DayOfWeek)(nil)).Elem()
}

type DeviceNotSupportedReason string

const (
	DeviceNotSupportedReasonHost  = DeviceNotSupportedReason("host")
	DeviceNotSupportedReasonGuest = DeviceNotSupportedReason("guest")
)

func init() {
	t["DeviceNotSupportedReason"] = reflect.TypeOf((*DeviceNotSupportedReason)(nil)).Elem()
}

type DiagnosticManagerLogCreator string

const (
	DiagnosticManagerLogCreatorVpxd      = DiagnosticManagerLogCreator("vpxd")
	DiagnosticManagerLogCreatorVpxa      = DiagnosticManagerLogCreator("vpxa")
	DiagnosticManagerLogCreatorHostd     = DiagnosticManagerLogCreator("hostd")
	DiagnosticManagerLogCreatorServerd   = DiagnosticManagerLogCreator("serverd")
	DiagnosticManagerLogCreatorInstall   = DiagnosticManagerLogCreator("install")
	DiagnosticManagerLogCreatorVpxClient = DiagnosticManagerLogCreator("vpxClient")
	DiagnosticManagerLogCreatorRecordLog = DiagnosticManagerLogCreator("recordLog")
)

func init() {
	t["DiagnosticManagerLogCreator"] = reflect.TypeOf((*DiagnosticManagerLogCreator)(nil)).Elem()
}

type DiagnosticManagerLogFormat string

const (
	DiagnosticManagerLogFormatPlain = DiagnosticManagerLogFormat("plain")
)

func init() {
	t["DiagnosticManagerLogFormat"] = reflect.TypeOf((*DiagnosticManagerLogFormat)(nil)).Elem()
}

type DiagnosticPartitionStorageType string

const (
	DiagnosticPartitionStorageTypeDirectAttached  = DiagnosticPartitionStorageType("directAttached")
	DiagnosticPartitionStorageTypeNetworkAttached = DiagnosticPartitionStorageType("networkAttached")
)

func init() {
	t["DiagnosticPartitionStorageType"] = reflect.TypeOf((*DiagnosticPartitionStorageType)(nil)).Elem()
}

type DiagnosticPartitionType string

const (
	DiagnosticPartitionTypeSingleHost = DiagnosticPartitionType("singleHost")
	DiagnosticPartitionTypeMultiHost  = DiagnosticPartitionType("multiHost")
)

func init() {
	t["DiagnosticPartitionType"] = reflect.TypeOf((*DiagnosticPartitionType)(nil)).Elem()
}

type DisallowedChangeByServiceDisallowedChange string

const (
	DisallowedChangeByServiceDisallowedChangeHotExtendDisk = DisallowedChangeByServiceDisallowedChange("hotExtendDisk")
)

func init() {
	t["DisallowedChangeByServiceDisallowedChange"] = reflect.TypeOf((*DisallowedChangeByServiceDisallowedChange)(nil)).Elem()
}

type DistributedVirtualPortgroupMetaTagName string

const (
	DistributedVirtualPortgroupMetaTagNameDvsName       = DistributedVirtualPortgroupMetaTagName("dvsName")
	DistributedVirtualPortgroupMetaTagNamePortgroupName = DistributedVirtualPortgroupMetaTagName("portgroupName")
	DistributedVirtualPortgroupMetaTagNamePortIndex     = DistributedVirtualPortgroupMetaTagName("portIndex")
)

func init() {
	t["DistributedVirtualPortgroupMetaTagName"] = reflect.TypeOf((*DistributedVirtualPortgroupMetaTagName)(nil)).Elem()
}

type DistributedVirtualPortgroupPortgroupType string

const (
	DistributedVirtualPortgroupPortgroupTypeEarlyBinding = DistributedVirtualPortgroupPortgroupType("earlyBinding")
	DistributedVirtualPortgroupPortgroupTypeLateBinding  = DistributedVirtualPortgroupPortgroupType("lateBinding")
	DistributedVirtualPortgroupPortgroupTypeEphemeral    = DistributedVirtualPortgroupPortgroupType("ephemeral")
)

func init() {
	t["DistributedVirtualPortgroupPortgroupType"] = reflect.TypeOf((*DistributedVirtualPortgroupPortgroupType)(nil)).Elem()
}

type DistributedVirtualSwitchHostInfrastructureTrafficClass string

const (
	DistributedVirtualSwitchHostInfrastructureTrafficClassManagement     = DistributedVirtualSwitchHostInfrastructureTrafficClass("management")
	DistributedVirtualSwitchHostInfrastructureTrafficClassFaultTolerance = DistributedVirtualSwitchHostInfrastructureTrafficClass("faultTolerance")
	DistributedVirtualSwitchHostInfrastructureTrafficClassVmotion        = DistributedVirtualSwitchHostInfrastructureTrafficClass("vmotion")
	DistributedVirtualSwitchHostInfrastructureTrafficClassVirtualMachine = DistributedVirtualSwitchHostInfrastructureTrafficClass("virtualMachine")
	DistributedVirtualSwitchHostInfrastructureTrafficClassISCSI          = DistributedVirtualSwitchHostInfrastructureTrafficClass("iSCSI")
	DistributedVirtualSwitchHostInfrastructureTrafficClassNfs            = DistributedVirtualSwitchHostInfrastructureTrafficClass("nfs")
	DistributedVirtualSwitchHostInfrastructureTrafficClassHbr            = DistributedVirtualSwitchHostInfrastructureTrafficClass("hbr")
	DistributedVirtualSwitchHostInfrastructureTrafficClassVsan           = DistributedVirtualSwitchHostInfrastructureTrafficClass("vsan")
	DistributedVirtualSwitchHostInfrastructureTrafficClassVdp            = DistributedVirtualSwitchHostInfrastructureTrafficClass("vdp")
)

func init() {
	t["DistributedVirtualSwitchHostInfrastructureTrafficClass"] = reflect.TypeOf((*DistributedVirtualSwitchHostInfrastructureTrafficClass)(nil)).Elem()
}

type DistributedVirtualSwitchHostMemberHostComponentState string

const (
	DistributedVirtualSwitchHostMemberHostComponentStateUp           = DistributedVirtualSwitchHostMemberHostComponentState("up")
	DistributedVirtualSwitchHostMemberHostComponentStatePending      = DistributedVirtualSwitchHostMemberHostComponentState("pending")
	DistributedVirtualSwitchHostMemberHostComponentStateOutOfSync    = DistributedVirtualSwitchHostMemberHostComponentState("outOfSync")
	DistributedVirtualSwitchHostMemberHostComponentStateWarning      = DistributedVirtualSwitchHostMemberHostComponentState("warning")
	DistributedVirtualSwitchHostMemberHostComponentStateDisconnected = DistributedVirtualSwitchHostMemberHostComponentState("disconnected")
	DistributedVirtualSwitchHostMemberHostComponentStateDown         = DistributedVirtualSwitchHostMemberHostComponentState("down")
)

func init() {
	t["DistributedVirtualSwitchHostMemberHostComponentState"] = reflect.TypeOf((*DistributedVirtualSwitchHostMemberHostComponentState)(nil)).Elem()
}

type DistributedVirtualSwitchNetworkResourceControlVersion string

const (
	DistributedVirtualSwitchNetworkResourceControlVersionVersion2 = DistributedVirtualSwitchNetworkResourceControlVersion("version2")
	DistributedVirtualSwitchNetworkResourceControlVersionVersion3 = DistributedVirtualSwitchNetworkResourceControlVersion("version3")
)

func init() {
	t["DistributedVirtualSwitchNetworkResourceControlVersion"] = reflect.TypeOf((*DistributedVirtualSwitchNetworkResourceControlVersion)(nil)).Elem()
}

type DistributedVirtualSwitchNicTeamingPolicyMode string

const (
	DistributedVirtualSwitchNicTeamingPolicyModeLoadbalance_ip        = DistributedVirtualSwitchNicTeamingPolicyMode("loadbalance_ip")
	DistributedVirtualSwitchNicTeamingPolicyModeLoadbalance_srcmac    = DistributedVirtualSwitchNicTeamingPolicyMode("loadbalance_srcmac")
	DistributedVirtualSwitchNicTeamingPolicyModeLoadbalance_srcid     = DistributedVirtualSwitchNicTeamingPolicyMode("loadbalance_srcid")
	DistributedVirtualSwitchNicTeamingPolicyModeFailover_explicit     = DistributedVirtualSwitchNicTeamingPolicyMode("failover_explicit")
	DistributedVirtualSwitchNicTeamingPolicyModeLoadbalance_loadbased = DistributedVirtualSwitchNicTeamingPolicyMode("loadbalance_loadbased")
)

func init() {
	t["DistributedVirtualSwitchNicTeamingPolicyMode"] = reflect.TypeOf((*DistributedVirtualSwitchNicTeamingPolicyMode)(nil)).Elem()
}

type DistributedVirtualSwitchPortConnecteeConnecteeType string

const (
	DistributedVirtualSwitchPortConnecteeConnecteeTypePnic            = DistributedVirtualSwitchPortConnecteeConnecteeType("pnic")
	DistributedVirtualSwitchPortConnecteeConnecteeTypeVmVnic          = DistributedVirtualSwitchPortConnecteeConnecteeType("vmVnic")
	DistributedVirtualSwitchPortConnecteeConnecteeTypeHostConsoleVnic = DistributedVirtualSwitchPortConnecteeConnecteeType("hostConsoleVnic")
	DistributedVirtualSwitchPortConnecteeConnecteeTypeHostVmkVnic     = DistributedVirtualSwitchPortConnecteeConnecteeType("hostVmkVnic")
)

func init() {
	t["DistributedVirtualSwitchPortConnecteeConnecteeType"] = reflect.TypeOf((*DistributedVirtualSwitchPortConnecteeConnecteeType)(nil)).Elem()
}

type DistributedVirtualSwitchProductSpecOperationType string

const (
	DistributedVirtualSwitchProductSpecOperationTypePreInstall             = DistributedVirtualSwitchProductSpecOperationType("preInstall")
	DistributedVirtualSwitchProductSpecOperationTypeUpgrade                = DistributedVirtualSwitchProductSpecOperationType("upgrade")
	DistributedVirtualSwitchProductSpecOperationTypeNotifyAvailableUpgrade = DistributedVirtualSwitchProductSpecOperationType("notifyAvailableUpgrade")
	DistributedVirtualSwitchProductSpecOperationTypeProceedWithUpgrade     = DistributedVirtualSwitchProductSpecOperationType("proceedWithUpgrade")
	DistributedVirtualSwitchProductSpecOperationTypeUpdateBundleInfo       = DistributedVirtualSwitchProductSpecOperationType("updateBundleInfo")
)

func init() {
	t["DistributedVirtualSwitchProductSpecOperationType"] = reflect.TypeOf((*DistributedVirtualSwitchProductSpecOperationType)(nil)).Elem()
}

type DpmBehavior string

const (
	DpmBehaviorManual    = DpmBehavior("manual")
	DpmBehaviorAutomated = DpmBehavior("automated")
)

func init() {
	t["DpmBehavior"] = reflect.TypeOf((*DpmBehavior)(nil)).Elem()
}

type DrsBehavior string

const (
	DrsBehaviorManual             = DrsBehavior("manual")
	DrsBehaviorPartiallyAutomated = DrsBehavior("partiallyAutomated")
	DrsBehaviorFullyAutomated     = DrsBehavior("fullyAutomated")
)

func init() {
	t["DrsBehavior"] = reflect.TypeOf((*DrsBehavior)(nil)).Elem()
}

type DrsInjectorWorkloadCorrelationState string

const (
	DrsInjectorWorkloadCorrelationStateCorrelated   = DrsInjectorWorkloadCorrelationState("Correlated")
	DrsInjectorWorkloadCorrelationStateUncorrelated = DrsInjectorWorkloadCorrelationState("Uncorrelated")
)

func init() {
	t["DrsInjectorWorkloadCorrelationState"] = reflect.TypeOf((*DrsInjectorWorkloadCorrelationState)(nil)).Elem()
}

type DrsRecommendationReasonCode string

const (
	DrsRecommendationReasonCodeFairnessCpuAvg = DrsRecommendationReasonCode("fairnessCpuAvg")
	DrsRecommendationReasonCodeFairnessMemAvg = DrsRecommendationReasonCode("fairnessMemAvg")
	DrsRecommendationReasonCodeJointAffin     = DrsRecommendationReasonCode("jointAffin")
	DrsRecommendationReasonCodeAntiAffin      = DrsRecommendationReasonCode("antiAffin")
	DrsRecommendationReasonCodeHostMaint      = DrsRecommendationReasonCode("hostMaint")
)

func init() {
	t["DrsRecommendationReasonCode"] = reflect.TypeOf((*DrsRecommendationReasonCode)(nil)).Elem()
}

type DvsEventPortBlockState string

const (
	DvsEventPortBlockStateUnset     = DvsEventPortBlockState("unset")
	DvsEventPortBlockStateBlocked   = DvsEventPortBlockState("blocked")
	DvsEventPortBlockStateUnblocked = DvsEventPortBlockState("unblocked")
	DvsEventPortBlockStateUnknown   = DvsEventPortBlockState("unknown")
)

func init() {
	t["DvsEventPortBlockState"] = reflect.TypeOf((*DvsEventPortBlockState)(nil)).Elem()
}

type DvsFilterOnFailure string

const (
	DvsFilterOnFailureFailOpen   = DvsFilterOnFailure("failOpen")
	DvsFilterOnFailureFailClosed = DvsFilterOnFailure("failClosed")
)

func init() {
	t["DvsFilterOnFailure"] = reflect.TypeOf((*DvsFilterOnFailure)(nil)).Elem()
}

type DvsNetworkRuleDirectionType string

const (
	DvsNetworkRuleDirectionTypeIncomingPackets = DvsNetworkRuleDirectionType("incomingPackets")
	DvsNetworkRuleDirectionTypeOutgoingPackets = DvsNetworkRuleDirectionType("outgoingPackets")
	DvsNetworkRuleDirectionTypeBoth            = DvsNetworkRuleDirectionType("both")
)

func init() {
	t["DvsNetworkRuleDirectionType"] = reflect.TypeOf((*DvsNetworkRuleDirectionType)(nil)).Elem()
}

type EntityImportType string

const (
	EntityImportTypeCreateEntityWithNewIdentifier      = EntityImportType("createEntityWithNewIdentifier")
	EntityImportTypeCreateEntityWithOriginalIdentifier = EntityImportType("createEntityWithOriginalIdentifier")
	EntityImportTypeApplyToEntitySpecified             = EntityImportType("applyToEntitySpecified")
)

func init() {
	t["EntityImportType"] = reflect.TypeOf((*EntityImportType)(nil)).Elem()
}

type EntityType string

const (
	EntityTypeDistributedVirtualSwitch    = EntityType("distributedVirtualSwitch")
	EntityTypeDistributedVirtualPortgroup = EntityType("distributedVirtualPortgroup")
)

func init() {
	t["EntityType"] = reflect.TypeOf((*EntityType)(nil)).Elem()
}

type EventAlarmExpressionComparisonOperator string

const (
	EventAlarmExpressionComparisonOperatorEquals           = EventAlarmExpressionComparisonOperator("equals")
	EventAlarmExpressionComparisonOperatorNotEqualTo       = EventAlarmExpressionComparisonOperator("notEqualTo")
	EventAlarmExpressionComparisonOperatorStartsWith       = EventAlarmExpressionComparisonOperator("startsWith")
	EventAlarmExpressionComparisonOperatorDoesNotStartWith = EventAlarmExpressionComparisonOperator("doesNotStartWith")
	EventAlarmExpressionComparisonOperatorEndsWith         = EventAlarmExpressionComparisonOperator("endsWith")
	EventAlarmExpressionComparisonOperatorDoesNotEndWith   = EventAlarmExpressionComparisonOperator("doesNotEndWith")
)

func init() {
	t["EventAlarmExpressionComparisonOperator"] = reflect.TypeOf((*EventAlarmExpressionComparisonOperator)(nil)).Elem()
}

type EventCategory string

const (
	EventCategoryInfo    = EventCategory("info")
	EventCategoryWarning = EventCategory("warning")
	EventCategoryError   = EventCategory("error")
	EventCategoryUser    = EventCategory("user")
)

func init() {
	t["EventCategory"] = reflect.TypeOf((*EventCategory)(nil)).Elem()
}

type EventEventSeverity string

const (
	EventEventSeverityError   = EventEventSeverity("error")
	EventEventSeverityWarning = EventEventSeverity("warning")
	EventEventSeverityInfo    = EventEventSeverity("info")
	EventEventSeverityUser    = EventEventSeverity("user")
)

func init() {
	t["EventEventSeverity"] = reflect.TypeOf((*EventEventSeverity)(nil)).Elem()
}

type EventFilterSpecRecursionOption string

const (
	EventFilterSpecRecursionOptionSelf     = EventFilterSpecRecursionOption("self")
	EventFilterSpecRecursionOptionChildren = EventFilterSpecRecursionOption("children")
	EventFilterSpecRecursionOptionAll      = EventFilterSpecRecursionOption("all")
)

func init() {
	t["EventFilterSpecRecursionOption"] = reflect.TypeOf((*EventFilterSpecRecursionOption)(nil)).Elem()
}

type FibreChannelPortType string

const (
	FibreChannelPortTypeFabric       = FibreChannelPortType("fabric")
	FibreChannelPortTypeLoop         = FibreChannelPortType("loop")
	FibreChannelPortTypePointToPoint = FibreChannelPortType("pointToPoint")
	FibreChannelPortTypeUnknown      = FibreChannelPortType("unknown")
)

func init() {
	t["FibreChannelPortType"] = reflect.TypeOf((*FibreChannelPortType)(nil)).Elem()
}

type FileSystemMountInfoVStorageSupportStatus string

const (
	FileSystemMountInfoVStorageSupportStatusVStorageSupported   = FileSystemMountInfoVStorageSupportStatus("vStorageSupported")
	FileSystemMountInfoVStorageSupportStatusVStorageUnsupported = FileSystemMountInfoVStorageSupportStatus("vStorageUnsupported")
	FileSystemMountInfoVStorageSupportStatusVStorageUnknown     = FileSystemMountInfoVStorageSupportStatus("vStorageUnknown")
)

func init() {
	t["FileSystemMountInfoVStorageSupportStatus"] = reflect.TypeOf((*FileSystemMountInfoVStorageSupportStatus)(nil)).Elem()
}

type FtIssuesOnHostHostSelectionType string

const (
	FtIssuesOnHostHostSelectionTypeUser = FtIssuesOnHostHostSelectionType("user")
	FtIssuesOnHostHostSelectionTypeVc   = FtIssuesOnHostHostSelectionType("vc")
	FtIssuesOnHostHostSelectionTypeDrs  = FtIssuesOnHostHostSelectionType("drs")
)

func init() {
	t["FtIssuesOnHostHostSelectionType"] = reflect.TypeOf((*FtIssuesOnHostHostSelectionType)(nil)).Elem()
}

type GuestFileType string

const (
	GuestFileTypeFile      = GuestFileType("file")
	GuestFileTypeDirectory = GuestFileType("directory")
	GuestFileTypeSymlink   = GuestFileType("symlink")
)

func init() {
	t["GuestFileType"] = reflect.TypeOf((*GuestFileType)(nil)).Elem()
}

type GuestInfoAppStateType string

const (
	GuestInfoAppStateTypeNone              = GuestInfoAppStateType("none")
	GuestInfoAppStateTypeAppStateOk        = GuestInfoAppStateType("appStateOk")
	GuestInfoAppStateTypeAppStateNeedReset = GuestInfoAppStateType("appStateNeedReset")
)

func init() {
	t["GuestInfoAppStateType"] = reflect.TypeOf((*GuestInfoAppStateType)(nil)).Elem()
}

type GuestOsDescriptorFirmwareType string

const (
	GuestOsDescriptorFirmwareTypeBios = GuestOsDescriptorFirmwareType("bios")
	GuestOsDescriptorFirmwareTypeEfi  = GuestOsDescriptorFirmwareType("efi")
)

func init() {
	t["GuestOsDescriptorFirmwareType"] = reflect.TypeOf((*GuestOsDescriptorFirmwareType)(nil)).Elem()
}

type GuestOsDescriptorSupportLevel string

const (
	GuestOsDescriptorSupportLevelExperimental = GuestOsDescriptorSupportLevel("experimental")
	GuestOsDescriptorSupportLevelLegacy       = GuestOsDescriptorSupportLevel("legacy")
	GuestOsDescriptorSupportLevelTerminated   = GuestOsDescriptorSupportLevel("terminated")
	GuestOsDescriptorSupportLevelSupported    = GuestOsDescriptorSupportLevel("supported")
	GuestOsDescriptorSupportLevelUnsupported  = GuestOsDescriptorSupportLevel("unsupported")
	GuestOsDescriptorSupportLevelDeprecated   = GuestOsDescriptorSupportLevel("deprecated")
	GuestOsDescriptorSupportLevelTechPreview  = GuestOsDescriptorSupportLevel("techPreview")
)

func init() {
	t["GuestOsDescriptorSupportLevel"] = reflect.TypeOf((*GuestOsDescriptorSupportLevel)(nil)).Elem()
}

type GuestRegKeyWowSpec string

const (
	GuestRegKeyWowSpecWOWNative = GuestRegKeyWowSpec("WOWNative")
	GuestRegKeyWowSpecWOW32     = GuestRegKeyWowSpec("WOW32")
	GuestRegKeyWowSpecWOW64     = GuestRegKeyWowSpec("WOW64")
)

func init() {
	t["GuestRegKeyWowSpec"] = reflect.TypeOf((*GuestRegKeyWowSpec)(nil)).Elem()
}

type HealthUpdateInfoComponentType string

const (
	HealthUpdateInfoComponentTypeMemory  = HealthUpdateInfoComponentType("Memory")
	HealthUpdateInfoComponentTypePower   = HealthUpdateInfoComponentType("Power")
	HealthUpdateInfoComponentTypeFan     = HealthUpdateInfoComponentType("Fan")
	HealthUpdateInfoComponentTypeNetwork = HealthUpdateInfoComponentType("Network")
	HealthUpdateInfoComponentTypeStorage = HealthUpdateInfoComponentType("Storage")
)

func init() {
	t["HealthUpdateInfoComponentType"] = reflect.TypeOf((*HealthUpdateInfoComponentType)(nil)).Elem()
}

type HostAccessMode string

const (
	HostAccessModeAccessNone     = HostAccessMode("accessNone")
	HostAccessModeAccessAdmin    = HostAccessMode("accessAdmin")
	HostAccessModeAccessNoAccess = HostAccessMode("accessNoAccess")
	HostAccessModeAccessReadOnly = HostAccessMode("accessReadOnly")
	HostAccessModeAccessOther    = HostAccessMode("accessOther")
)

func init() {
	t["HostAccessMode"] = reflect.TypeOf((*HostAccessMode)(nil)).Elem()
}

type HostActiveDirectoryAuthenticationCertificateDigest string

const (
	HostActiveDirectoryAuthenticationCertificateDigestSHA1 = HostActiveDirectoryAuthenticationCertificateDigest("SHA1")
)

func init() {
	t["HostActiveDirectoryAuthenticationCertificateDigest"] = reflect.TypeOf((*HostActiveDirectoryAuthenticationCertificateDigest)(nil)).Elem()
}

type HostActiveDirectoryInfoDomainMembershipStatus string

const (
	HostActiveDirectoryInfoDomainMembershipStatusUnknown           = HostActiveDirectoryInfoDomainMembershipStatus("unknown")
	HostActiveDirectoryInfoDomainMembershipStatusOk                = HostActiveDirectoryInfoDomainMembershipStatus("ok")
	HostActiveDirectoryInfoDomainMembershipStatusNoServers         = HostActiveDirectoryInfoDomainMembershipStatus("noServers")
	HostActiveDirectoryInfoDomainMembershipStatusClientTrustBroken = HostActiveDirectoryInfoDomainMembershipStatus("clientTrustBroken")
	HostActiveDirectoryInfoDomainMembershipStatusServerTrustBroken = HostActiveDirectoryInfoDomainMembershipStatus("serverTrustBroken")
	HostActiveDirectoryInfoDomainMembershipStatusInconsistentTrust = HostActiveDirectoryInfoDomainMembershipStatus("inconsistentTrust")
	HostActiveDirectoryInfoDomainMembershipStatusOtherProblem      = HostActiveDirectoryInfoDomainMembershipStatus("otherProblem")
)

func init() {
	t["HostActiveDirectoryInfoDomainMembershipStatus"] = reflect.TypeOf((*HostActiveDirectoryInfoDomainMembershipStatus)(nil)).Elem()
}

type HostCapabilityFtUnsupportedReason string

const (
	HostCapabilityFtUnsupportedReasonVMotionNotLicensed  = HostCapabilityFtUnsupportedReason("vMotionNotLicensed")
	HostCapabilityFtUnsupportedReasonMissingVMotionNic   = HostCapabilityFtUnsupportedReason("missingVMotionNic")
	HostCapabilityFtUnsupportedReasonMissingFTLoggingNic = HostCapabilityFtUnsupportedReason("missingFTLoggingNic")
	HostCapabilityFtUnsupportedReasonFtNotLicensed       = HostCapabilityFtUnsupportedReason("ftNotLicensed")
	HostCapabilityFtUnsupportedReasonHaAgentIssue        = HostCapabilityFtUnsupportedReason("haAgentIssue")
	HostCapabilityFtUnsupportedReasonUnsupportedProduct  = HostCapabilityFtUnsupportedReason("unsupportedProduct")
	HostCapabilityFtUnsupportedReasonCpuHvUnsupported    = HostCapabilityFtUnsupportedReason("cpuHvUnsupported")
	HostCapabilityFtUnsupportedReasonCpuHwmmuUnsupported = HostCapabilityFtUnsupportedReason("cpuHwmmuUnsupported")
	HostCapabilityFtUnsupportedReasonCpuHvDisabled       = HostCapabilityFtUnsupportedReason("cpuHvDisabled")
)

func init() {
	t["HostCapabilityFtUnsupportedReason"] = reflect.TypeOf((*HostCapabilityFtUnsupportedReason)(nil)).Elem()
}

type HostCapabilityVmDirectPathGen2UnsupportedReason string

const (
	HostCapabilityVmDirectPathGen2UnsupportedReasonHostNptIncompatibleProduct  = HostCapabilityVmDirectPathGen2UnsupportedReason("hostNptIncompatibleProduct")
	HostCapabilityVmDirectPathGen2UnsupportedReasonHostNptIncompatibleHardware = HostCapabilityVmDirectPathGen2UnsupportedReason("hostNptIncompatibleHardware")
	HostCapabilityVmDirectPathGen2UnsupportedReasonHostNptDisabled             = HostCapabilityVmDirectPathGen2UnsupportedReason("hostNptDisabled")
)

func init() {
	t["HostCapabilityVmDirectPathGen2UnsupportedReason"] = reflect.TypeOf((*HostCapabilityVmDirectPathGen2UnsupportedReason)(nil)).Elem()
}

type HostCertificateManagerCertificateInfoCertificateStatus string

const (
	HostCertificateManagerCertificateInfoCertificateStatusUnknown            = HostCertificateManagerCertificateInfoCertificateStatus("unknown")
	HostCertificateManagerCertificateInfoCertificateStatusExpired            = HostCertificateManagerCertificateInfoCertificateStatus("expired")
	HostCertificateManagerCertificateInfoCertificateStatusExpiring           = HostCertificateManagerCertificateInfoCertificateStatus("expiring")
	HostCertificateManagerCertificateInfoCertificateStatusExpiringShortly    = HostCertificateManagerCertificateInfoCertificateStatus("expiringShortly")
	HostCertificateManagerCertificateInfoCertificateStatusExpirationImminent = HostCertificateManagerCertificateInfoCertificateStatus("expirationImminent")
	HostCertificateManagerCertificateInfoCertificateStatusGood               = HostCertificateManagerCertificateInfoCertificateStatus("good")
)

func init() {
	t["HostCertificateManagerCertificateInfoCertificateStatus"] = reflect.TypeOf((*HostCertificateManagerCertificateInfoCertificateStatus)(nil)).Elem()
}

type HostConfigChangeMode string

const (
	HostConfigChangeModeModify  = HostConfigChangeMode("modify")
	HostConfigChangeModeReplace = HostConfigChangeMode("replace")
)

func init() {
	t["HostConfigChangeMode"] = reflect.TypeOf((*HostConfigChangeMode)(nil)).Elem()
}

type HostConfigChangeOperation string

const (
	HostConfigChangeOperationAdd    = HostConfigChangeOperation("add")
	HostConfigChangeOperationRemove = HostConfigChangeOperation("remove")
	HostConfigChangeOperationEdit   = HostConfigChangeOperation("edit")
	HostConfigChangeOperationIgnore = HostConfigChangeOperation("ignore")
)

func init() {
	t["HostConfigChangeOperation"] = reflect.TypeOf((*HostConfigChangeOperation)(nil)).Elem()
}

type HostCpuPackageVendor string

const (
	HostCpuPackageVendorUnknown = HostCpuPackageVendor("unknown")
	HostCpuPackageVendorIntel   = HostCpuPackageVendor("intel")
	HostCpuPackageVendorAmd     = HostCpuPackageVendor("amd")
)

func init() {
	t["HostCpuPackageVendor"] = reflect.TypeOf((*HostCpuPackageVendor)(nil)).Elem()
}

type HostCpuPowerManagementInfoPolicyType string

const (
	HostCpuPowerManagementInfoPolicyTypeOff           = HostCpuPowerManagementInfoPolicyType("off")
	HostCpuPowerManagementInfoPolicyTypeStaticPolicy  = HostCpuPowerManagementInfoPolicyType("staticPolicy")
	HostCpuPowerManagementInfoPolicyTypeDynamicPolicy = HostCpuPowerManagementInfoPolicyType("dynamicPolicy")
)

func init() {
	t["HostCpuPowerManagementInfoPolicyType"] = reflect.TypeOf((*HostCpuPowerManagementInfoPolicyType)(nil)).Elem()
}

type HostCryptoState string

const (
	HostCryptoStateIncapable = HostCryptoState("incapable")
	HostCryptoStatePrepared  = HostCryptoState("prepared")
	HostCryptoStateSafe      = HostCryptoState("safe")
)

func init() {
	t["HostCryptoState"] = reflect.TypeOf((*HostCryptoState)(nil)).Elem()
}

type HostDasErrorEventHostDasErrorReason string

const (
	HostDasErrorEventHostDasErrorReasonConfigFailed               = HostDasErrorEventHostDasErrorReason("configFailed")
	HostDasErrorEventHostDasErrorReasonTimeout                    = HostDasErrorEventHostDasErrorReason("timeout")
	HostDasErrorEventHostDasErrorReasonCommunicationInitFailed    = HostDasErrorEventHostDasErrorReason("communicationInitFailed")
	HostDasErrorEventHostDasErrorReasonHealthCheckScriptFailed    = HostDasErrorEventHostDasErrorReason("healthCheckScriptFailed")
	HostDasErrorEventHostDasErrorReasonAgentFailed                = HostDasErrorEventHostDasErrorReason("agentFailed")
	HostDasErrorEventHostDasErrorReasonAgentShutdown              = HostDasErrorEventHostDasErrorReason("agentShutdown")
	HostDasErrorEventHostDasErrorReasonIsolationAddressUnpingable = HostDasErrorEventHostDasErrorReason("isolationAddressUnpingable")
	HostDasErrorEventHostDasErrorReasonOther                      = HostDasErrorEventHostDasErrorReason("other")
)

func init() {
	t["HostDasErrorEventHostDasErrorReason"] = reflect.TypeOf((*HostDasErrorEventHostDasErrorReason)(nil)).Elem()
}

type HostDigestInfoDigestMethodType string

const (
	HostDigestInfoDigestMethodTypeSHA1 = HostDigestInfoDigestMethodType("SHA1")
	HostDigestInfoDigestMethodTypeMD5  = HostDigestInfoDigestMethodType("MD5")
)

func init() {
	t["HostDigestInfoDigestMethodType"] = reflect.TypeOf((*HostDigestInfoDigestMethodType)(nil)).Elem()
}

type HostDisconnectedEventReasonCode string

const (
	HostDisconnectedEventReasonCodeSslThumbprintVerifyFailed = HostDisconnectedEventReasonCode("sslThumbprintVerifyFailed")
	HostDisconnectedEventReasonCodeLicenseExpired            = HostDisconnectedEventReasonCode("licenseExpired")
	HostDisconnectedEventReasonCodeAgentUpgrade              = HostDisconnectedEventReasonCode("agentUpgrade")
	HostDisconnectedEventReasonCodeUserRequest               = HostDisconnectedEventReasonCode("userRequest")
	HostDisconnectedEventReasonCodeInsufficientLicenses      = HostDisconnectedEventReasonCode("insufficientLicenses")
	HostDisconnectedEventReasonCodeAgentOutOfDate            = HostDisconnectedEventReasonCode("agentOutOfDate")
	HostDisconnectedEventReasonCodePasswordDecryptFailure    = HostDisconnectedEventReasonCode("passwordDecryptFailure")
	HostDisconnectedEventReasonCodeUnknown                   = HostDisconnectedEventReasonCode("unknown")
	HostDisconnectedEventReasonCodeVcVRAMCapacityExceeded    = HostDisconnectedEventReasonCode("vcVRAMCapacityExceeded")
)

func init() {
	t["HostDisconnectedEventReasonCode"] = reflect.TypeOf((*HostDisconnectedEventReasonCode)(nil)).Elem()
}

type HostDiskPartitionInfoPartitionFormat string

const (
	HostDiskPartitionInfoPartitionFormatGpt     = HostDiskPartitionInfoPartitionFormat("gpt")
	HostDiskPartitionInfoPartitionFormatMbr     = HostDiskPartitionInfoPartitionFormat("mbr")
	HostDiskPartitionInfoPartitionFormatUnknown = HostDiskPartitionInfoPartitionFormat("unknown")
)

func init() {
	t["HostDiskPartitionInfoPartitionFormat"] = reflect.TypeOf((*HostDiskPartitionInfoPartitionFormat)(nil)).Elem()
}

type HostDiskPartitionInfoType string

const (
	HostDiskPartitionInfoTypeNone          = HostDiskPartitionInfoType("none")
	HostDiskPartitionInfoTypeVmfs          = HostDiskPartitionInfoType("vmfs")
	HostDiskPartitionInfoTypeLinuxNative   = HostDiskPartitionInfoType("linuxNative")
	HostDiskPartitionInfoTypeLinuxSwap     = HostDiskPartitionInfoType("linuxSwap")
	HostDiskPartitionInfoTypeExtended      = HostDiskPartitionInfoType("extended")
	HostDiskPartitionInfoTypeNtfs          = HostDiskPartitionInfoType("ntfs")
	HostDiskPartitionInfoTypeVmkDiagnostic = HostDiskPartitionInfoType("vmkDiagnostic")
	HostDiskPartitionInfoTypeVffs          = HostDiskPartitionInfoType("vffs")
)

func init() {
	t["HostDiskPartitionInfoType"] = reflect.TypeOf((*HostDiskPartitionInfoType)(nil)).Elem()
}

type HostFeatureVersionKey string

const (
	HostFeatureVersionKeyFaultTolerance = HostFeatureVersionKey("faultTolerance")
)

func init() {
	t["HostFeatureVersionKey"] = reflect.TypeOf((*HostFeatureVersionKey)(nil)).Elem()
}

type HostFileSystemVolumeFileSystemType string

const (
	HostFileSystemVolumeFileSystemTypeVMFS  = HostFileSystemVolumeFileSystemType("VMFS")
	HostFileSystemVolumeFileSystemTypeNFS   = HostFileSystemVolumeFileSystemType("NFS")
	HostFileSystemVolumeFileSystemTypeNFS41 = HostFileSystemVolumeFileSystemType("NFS41")
	HostFileSystemVolumeFileSystemTypeCIFS  = HostFileSystemVolumeFileSystemType("CIFS")
	HostFileSystemVolumeFileSystemTypeVsan  = HostFileSystemVolumeFileSystemType("vsan")
	HostFileSystemVolumeFileSystemTypeVFFS  = HostFileSystemVolumeFileSystemType("VFFS")
	HostFileSystemVolumeFileSystemTypeVVOL  = HostFileSystemVolumeFileSystemType("VVOL")
	HostFileSystemVolumeFileSystemTypeOTHER = HostFileSystemVolumeFileSystemType("OTHER")
)

func init() {
	t["HostFileSystemVolumeFileSystemType"] = reflect.TypeOf((*HostFileSystemVolumeFileSystemType)(nil)).Elem()
}

type HostFirewallRuleDirection string

const (
	HostFirewallRuleDirectionInbound  = HostFirewallRuleDirection("inbound")
	HostFirewallRuleDirectionOutbound = HostFirewallRuleDirection("outbound")
)

func init() {
	t["HostFirewallRuleDirection"] = reflect.TypeOf((*HostFirewallRuleDirection)(nil)).Elem()
}

type HostFirewallRulePortType string

const (
	HostFirewallRulePortTypeSrc = HostFirewallRulePortType("src")
	HostFirewallRulePortTypeDst = HostFirewallRulePortType("dst")
)

func init() {
	t["HostFirewallRulePortType"] = reflect.TypeOf((*HostFirewallRulePortType)(nil)).Elem()
}

type HostFirewallRuleProtocol string

const (
	HostFirewallRuleProtocolTcp = HostFirewallRuleProtocol("tcp")
	HostFirewallRuleProtocolUdp = HostFirewallRuleProtocol("udp")
)

func init() {
	t["HostFirewallRuleProtocol"] = reflect.TypeOf((*HostFirewallRuleProtocol)(nil)).Elem()
}

type HostGraphicsConfigGraphicsType string

const (
	HostGraphicsConfigGraphicsTypeShared       = HostGraphicsConfigGraphicsType("shared")
	HostGraphicsConfigGraphicsTypeSharedDirect = HostGraphicsConfigGraphicsType("sharedDirect")
)

func init() {
	t["HostGraphicsConfigGraphicsType"] = reflect.TypeOf((*HostGraphicsConfigGraphicsType)(nil)).Elem()
}

type HostGraphicsConfigSharedPassthruAssignmentPolicy string

const (
	HostGraphicsConfigSharedPassthruAssignmentPolicyPerformance   = HostGraphicsConfigSharedPassthruAssignmentPolicy("performance")
	HostGraphicsConfigSharedPassthruAssignmentPolicyConsolidation = HostGraphicsConfigSharedPassthruAssignmentPolicy("consolidation")
)

func init() {
	t["HostGraphicsConfigSharedPassthruAssignmentPolicy"] = reflect.TypeOf((*HostGraphicsConfigSharedPassthruAssignmentPolicy)(nil)).Elem()
}

type HostGraphicsInfoGraphicsType string

const (
	HostGraphicsInfoGraphicsTypeBasic        = HostGraphicsInfoGraphicsType("basic")
	HostGraphicsInfoGraphicsTypeShared       = HostGraphicsInfoGraphicsType("shared")
	HostGraphicsInfoGraphicsTypeDirect       = HostGraphicsInfoGraphicsType("direct")
	HostGraphicsInfoGraphicsTypeSharedDirect = HostGraphicsInfoGraphicsType("sharedDirect")
)

func init() {
	t["HostGraphicsInfoGraphicsType"] = reflect.TypeOf((*HostGraphicsInfoGraphicsType)(nil)).Elem()
}

type HostHardwareElementStatus string

const (
	HostHardwareElementStatusUnknown = HostHardwareElementStatus("Unknown")
	HostHardwareElementStatusGreen   = HostHardwareElementStatus("Green")
	HostHardwareElementStatusYellow  = HostHardwareElementStatus("Yellow")
	HostHardwareElementStatusRed     = HostHardwareElementStatus("Red")
)

func init() {
	t["HostHardwareElementStatus"] = reflect.TypeOf((*HostHardwareElementStatus)(nil)).Elem()
}

type HostHasComponentFailureHostComponentType string

const (
	HostHasComponentFailureHostComponentTypeDatastore = HostHasComponentFailureHostComponentType("Datastore")
)

func init() {
	t["HostHasComponentFailureHostComponentType"] = reflect.TypeOf((*HostHasComponentFailureHostComponentType)(nil)).Elem()
}

type HostImageAcceptanceLevel string

const (
	HostImageAcceptanceLevelVmware_certified = HostImageAcceptanceLevel("vmware_certified")
	HostImageAcceptanceLevelVmware_accepted  = HostImageAcceptanceLevel("vmware_accepted")
	HostImageAcceptanceLevelPartner          = HostImageAcceptanceLevel("partner")
	HostImageAcceptanceLevelCommunity        = HostImageAcceptanceLevel("community")
)

func init() {
	t["HostImageAcceptanceLevel"] = reflect.TypeOf((*HostImageAcceptanceLevel)(nil)).Elem()
}

type HostIncompatibleForFaultToleranceReason string

const (
	HostIncompatibleForFaultToleranceReasonProduct   = HostIncompatibleForFaultToleranceReason("product")
	HostIncompatibleForFaultToleranceReasonProcessor = HostIncompatibleForFaultToleranceReason("processor")
)

func init() {
	t["HostIncompatibleForFaultToleranceReason"] = reflect.TypeOf((*HostIncompatibleForFaultToleranceReason)(nil)).Elem()
}

type HostIncompatibleForRecordReplayReason string

const (
	HostIncompatibleForRecordReplayReasonProduct   = HostIncompatibleForRecordReplayReason("product")
	HostIncompatibleForRecordReplayReasonProcessor = HostIncompatibleForRecordReplayReason("processor")
)

func init() {
	t["HostIncompatibleForRecordReplayReason"] = reflect.TypeOf((*HostIncompatibleForRecordReplayReason)(nil)).Elem()
}

type HostInternetScsiHbaChapAuthenticationType string

const (
	HostInternetScsiHbaChapAuthenticationTypeChapProhibited  = HostInternetScsiHbaChapAuthenticationType("chapProhibited")
	HostInternetScsiHbaChapAuthenticationTypeChapDiscouraged = HostInternetScsiHbaChapAuthenticationType("chapDiscouraged")
	HostInternetScsiHbaChapAuthenticationTypeChapPreferred   = HostInternetScsiHbaChapAuthenticationType("chapPreferred")
	HostInternetScsiHbaChapAuthenticationTypeChapRequired    = HostInternetScsiHbaChapAuthenticationType("chapRequired")
)

func init() {
	t["HostInternetScsiHbaChapAuthenticationType"] = reflect.TypeOf((*HostInternetScsiHbaChapAuthenticationType)(nil)).Elem()
}

type HostInternetScsiHbaDigestType string

const (
	HostInternetScsiHbaDigestTypeDigestProhibited  = HostInternetScsiHbaDigestType("digestProhibited")
	HostInternetScsiHbaDigestTypeDigestDiscouraged = HostInternetScsiHbaDigestType("digestDiscouraged")
	HostInternetScsiHbaDigestTypeDigestPreferred   = HostInternetScsiHbaDigestType("digestPreferred")
	HostInternetScsiHbaDigestTypeDigestRequired    = HostInternetScsiHbaDigestType("digestRequired")
)

func init() {
	t["HostInternetScsiHbaDigestType"] = reflect.TypeOf((*HostInternetScsiHbaDigestType)(nil)).Elem()
}

type HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationType string

const (
	HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationTypeDHCP           = HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationType("DHCP")
	HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationTypeAutoConfigured = HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationType("AutoConfigured")
	HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationTypeStatic         = HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationType("Static")
	HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationTypeOther          = HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationType("Other")
)

func init() {
	t["HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationType"] = reflect.TypeOf((*HostInternetScsiHbaIscsiIpv6AddressAddressConfigurationType)(nil)).Elem()
}

type HostInternetScsiHbaIscsiIpv6AddressIPv6AddressOperation string

const (
	HostInternetScsiHbaIscsiIpv6AddressIPv6AddressOperationAdd    = HostInternetScsiHbaIscsiIpv6AddressIPv6AddressOperation("add")
	HostInternetScsiHbaIscsiIpv6AddressIPv6AddressOperationRemove = HostInternetScsiHbaIscsiIpv6AddressIPv6AddressOperation("remove")
)

func init() {
	t["HostInternetScsiHbaIscsiIpv6AddressIPv6AddressOperation"] = reflect.TypeOf((*HostInternetScsiHbaIscsiIpv6AddressIPv6AddressOperation)(nil)).Elem()
}

type HostInternetScsiHbaNetworkBindingSupportType string

const (
	HostInternetScsiHbaNetworkBindingSupportTypeNotsupported = HostInternetScsiHbaNetworkBindingSupportType("notsupported")
	HostInternetScsiHbaNetworkBindingSupportTypeOptional     = HostInternetScsiHbaNetworkBindingSupportType("optional")
	HostInternetScsiHbaNetworkBindingSupportTypeRequired     = HostInternetScsiHbaNetworkBindingSupportType("required")
)

func init() {
	t["HostInternetScsiHbaNetworkBindingSupportType"] = reflect.TypeOf((*HostInternetScsiHbaNetworkBindingSupportType)(nil)).Elem()
}

type HostInternetScsiHbaStaticTargetTargetDiscoveryMethod string

const (
	HostInternetScsiHbaStaticTargetTargetDiscoveryMethodStaticMethod     = HostInternetScsiHbaStaticTargetTargetDiscoveryMethod("staticMethod")
	HostInternetScsiHbaStaticTargetTargetDiscoveryMethodSendTargetMethod = HostInternetScsiHbaStaticTargetTargetDiscoveryMethod("sendTargetMethod")
	HostInternetScsiHbaStaticTargetTargetDiscoveryMethodSlpMethod        = HostInternetScsiHbaStaticTargetTargetDiscoveryMethod("slpMethod")
	HostInternetScsiHbaStaticTargetTargetDiscoveryMethodIsnsMethod       = HostInternetScsiHbaStaticTargetTargetDiscoveryMethod("isnsMethod")
	HostInternetScsiHbaStaticTargetTargetDiscoveryMethodUnknownMethod    = HostInternetScsiHbaStaticTargetTargetDiscoveryMethod("unknownMethod")
)

func init() {
	t["HostInternetScsiHbaStaticTargetTargetDiscoveryMethod"] = reflect.TypeOf((*HostInternetScsiHbaStaticTargetTargetDiscoveryMethod)(nil)).Elem()
}

type HostIpConfigIpV6AddressConfigType string

const (
	HostIpConfigIpV6AddressConfigTypeOther     = HostIpConfigIpV6AddressConfigType("other")
	HostIpConfigIpV6AddressConfigTypeManual    = HostIpConfigIpV6AddressConfigType("manual")
	HostIpConfigIpV6AddressConfigTypeDhcp      = HostIpConfigIpV6AddressConfigType("dhcp")
	HostIpConfigIpV6AddressConfigTypeLinklayer = HostIpConfigIpV6AddressConfigType("linklayer")
	HostIpConfigIpV6AddressConfigTypeRandom    = HostIpConfigIpV6AddressConfigType("random")
)

func init() {
	t["HostIpConfigIpV6AddressConfigType"] = reflect.TypeOf((*HostIpConfigIpV6AddressConfigType)(nil)).Elem()
}

type HostIpConfigIpV6AddressStatus string

const (
	HostIpConfigIpV6AddressStatusPreferred    = HostIpConfigIpV6AddressStatus("preferred")
	HostIpConfigIpV6AddressStatusDeprecated   = HostIpConfigIpV6AddressStatus("deprecated")
	HostIpConfigIpV6AddressStatusInvalid      = HostIpConfigIpV6AddressStatus("invalid")
	HostIpConfigIpV6AddressStatusInaccessible = HostIpConfigIpV6AddressStatus("inaccessible")
	HostIpConfigIpV6AddressStatusUnknown      = HostIpConfigIpV6AddressStatus("unknown")
	HostIpConfigIpV6AddressStatusTentative    = HostIpConfigIpV6AddressStatus("tentative")
	HostIpConfigIpV6AddressStatusDuplicate    = HostIpConfigIpV6AddressStatus("duplicate")
)

func init() {
	t["HostIpConfigIpV6AddressStatus"] = reflect.TypeOf((*HostIpConfigIpV6AddressStatus)(nil)).Elem()
}

type HostLicensableResourceKey string

const (
	HostLicensableResourceKeyNumCpuPackages = HostLicensableResourceKey("numCpuPackages")
	HostLicensableResourceKeyNumCpuCores    = HostLicensableResourceKey("numCpuCores")
	HostLicensableResourceKeyMemorySize     = HostLicensableResourceKey("memorySize")
	HostLicensableResourceKeyMemoryForVms   = HostLicensableResourceKey("memoryForVms")
	HostLicensableResourceKeyNumVmsStarted  = HostLicensableResourceKey("numVmsStarted")
	HostLicensableResourceKeyNumVmsStarting = HostLicensableResourceKey("numVmsStarting")
)

func init() {
	t["HostLicensableResourceKey"] = reflect.TypeOf((*HostLicensableResourceKey)(nil)).Elem()
}

type HostLockdownMode string

const (
	HostLockdownModeLockdownDisabled = HostLockdownMode("lockdownDisabled")
	HostLockdownModeLockdownNormal   = HostLockdownMode("lockdownNormal")
	HostLockdownModeLockdownStrict   = HostLockdownMode("lockdownStrict")
)

func init() {
	t["HostLockdownMode"] = reflect.TypeOf((*HostLockdownMode)(nil)).Elem()
}

type HostLowLevelProvisioningManagerFileType string

const (
	HostLowLevelProvisioningManagerFileTypeFile        = HostLowLevelProvisioningManagerFileType("File")
	HostLowLevelProvisioningManagerFileTypeVirtualDisk = HostLowLevelProvisioningManagerFileType("VirtualDisk")
	HostLowLevelProvisioningManagerFileTypeDirectory   = HostLowLevelProvisioningManagerFileType("Directory")
)

func init() {
	t["HostLowLevelProvisioningManagerFileType"] = reflect.TypeOf((*HostLowLevelProvisioningManagerFileType)(nil)).Elem()
}

type HostLowLevelProvisioningManagerReloadTarget string

const (
	HostLowLevelProvisioningManagerReloadTargetCurrentConfig  = HostLowLevelProvisioningManagerReloadTarget("currentConfig")
	HostLowLevelProvisioningManagerReloadTargetSnapshotConfig = HostLowLevelProvisioningManagerReloadTarget("snapshotConfig")
)

func init() {
	t["HostLowLevelProvisioningManagerReloadTarget"] = reflect.TypeOf((*HostLowLevelProvisioningManagerReloadTarget)(nil)).Elem()
}

type HostMountInfoInaccessibleReason string

const (
	HostMountInfoInaccessibleReasonAllPathsDown_Start   = HostMountInfoInaccessibleReason("AllPathsDown_Start")
	HostMountInfoInaccessibleReasonAllPathsDown_Timeout = HostMountInfoInaccessibleReason("AllPathsDown_Timeout")
	HostMountInfoInaccessibleReasonPermanentDeviceLoss  = HostMountInfoInaccessibleReason("PermanentDeviceLoss")
)

func init() {
	t["HostMountInfoInaccessibleReason"] = reflect.TypeOf((*HostMountInfoInaccessibleReason)(nil)).Elem()
}

type HostMountMode string

const (
	HostMountModeReadWrite = HostMountMode("readWrite")
	HostMountModeReadOnly  = HostMountMode("readOnly")
)

func init() {
	t["HostMountMode"] = reflect.TypeOf((*HostMountMode)(nil)).Elem()
}

type HostNasVolumeSecurityType string

const (
	HostNasVolumeSecurityTypeAUTH_SYS  = HostNasVolumeSecurityType("AUTH_SYS")
	HostNasVolumeSecurityTypeSEC_KRB5  = HostNasVolumeSecurityType("SEC_KRB5")
	HostNasVolumeSecurityTypeSEC_KRB5I = HostNasVolumeSecurityType("SEC_KRB5I")
)

func init() {
	t["HostNasVolumeSecurityType"] = reflect.TypeOf((*HostNasVolumeSecurityType)(nil)).Elem()
}

type HostNetStackInstanceCongestionControlAlgorithmType string

const (
	HostNetStackInstanceCongestionControlAlgorithmTypeNewreno = HostNetStackInstanceCongestionControlAlgorithmType("newreno")
	HostNetStackInstanceCongestionControlAlgorithmTypeCubic   = HostNetStackInstanceCongestionControlAlgorithmType("cubic")
)

func init() {
	t["HostNetStackInstanceCongestionControlAlgorithmType"] = reflect.TypeOf((*HostNetStackInstanceCongestionControlAlgorithmType)(nil)).Elem()
}

type HostNetStackInstanceSystemStackKey string

const (
	HostNetStackInstanceSystemStackKeyDefaultTcpipStack   = HostNetStackInstanceSystemStackKey("defaultTcpipStack")
	HostNetStackInstanceSystemStackKeyVmotion             = HostNetStackInstanceSystemStackKey("vmotion")
	HostNetStackInstanceSystemStackKeyVSphereProvisioning = HostNetStackInstanceSystemStackKey("vSphereProvisioning")
)

func init() {
	t["HostNetStackInstanceSystemStackKey"] = reflect.TypeOf((*HostNetStackInstanceSystemStackKey)(nil)).Elem()
}

type HostNumericSensorHealthState string

const (
	HostNumericSensorHealthStateUnknown = HostNumericSensorHealthState("unknown")
	HostNumericSensorHealthStateGreen   = HostNumericSensorHealthState("green")
	HostNumericSensorHealthStateYellow  = HostNumericSensorHealthState("yellow")
	HostNumericSensorHealthStateRed     = HostNumericSensorHealthState("red")
)

func init() {
	t["HostNumericSensorHealthState"] = reflect.TypeOf((*HostNumericSensorHealthState)(nil)).Elem()
}

type HostNumericSensorType string

const (
	HostNumericSensorTypeFan         = HostNumericSensorType("fan")
	HostNumericSensorTypePower       = HostNumericSensorType("power")
	HostNumericSensorTypeTemperature = HostNumericSensorType("temperature")
	HostNumericSensorTypeVoltage     = HostNumericSensorType("voltage")
	HostNumericSensorTypeOther       = HostNumericSensorType("other")
	HostNumericSensorTypeProcessor   = HostNumericSensorType("processor")
	HostNumericSensorTypeMemory      = HostNumericSensorType("memory")
	HostNumericSensorTypeStorage     = HostNumericSensorType("storage")
	HostNumericSensorTypeSystemBoard = HostNumericSensorType("systemBoard")
	HostNumericSensorTypeBattery     = HostNumericSensorType("battery")
	HostNumericSensorTypeBios        = HostNumericSensorType("bios")
	HostNumericSensorTypeCable       = HostNumericSensorType("cable")
	HostNumericSensorTypeWatchdog    = HostNumericSensorType("watchdog")
)

func init() {
	t["HostNumericSensorType"] = reflect.TypeOf((*HostNumericSensorType)(nil)).Elem()
}

type HostOpaqueSwitchOpaqueSwitchState string

const (
	HostOpaqueSwitchOpaqueSwitchStateUp      = HostOpaqueSwitchOpaqueSwitchState("up")
	HostOpaqueSwitchOpaqueSwitchStateWarning = HostOpaqueSwitchOpaqueSwitchState("warning")
	HostOpaqueSwitchOpaqueSwitchStateDown    = HostOpaqueSwitchOpaqueSwitchState("down")
)

func init() {
	t["HostOpaqueSwitchOpaqueSwitchState"] = reflect.TypeOf((*HostOpaqueSwitchOpaqueSwitchState)(nil)).Elem()
}

type HostPatchManagerInstallState string

const (
	HostPatchManagerInstallStateHostRestarted = HostPatchManagerInstallState("hostRestarted")
	HostPatchManagerInstallStateImageActive   = HostPatchManagerInstallState("imageActive")
)

func init() {
	t["HostPatchManagerInstallState"] = reflect.TypeOf((*HostPatchManagerInstallState)(nil)).Elem()
}

type HostPatchManagerIntegrityStatus string

const (
	HostPatchManagerIntegrityStatusValidated           = HostPatchManagerIntegrityStatus("validated")
	HostPatchManagerIntegrityStatusKeyNotFound         = HostPatchManagerIntegrityStatus("keyNotFound")
	HostPatchManagerIntegrityStatusKeyRevoked          = HostPatchManagerIntegrityStatus("keyRevoked")
	HostPatchManagerIntegrityStatusKeyExpired          = HostPatchManagerIntegrityStatus("keyExpired")
	HostPatchManagerIntegrityStatusDigestMismatch      = HostPatchManagerIntegrityStatus("digestMismatch")
	HostPatchManagerIntegrityStatusNotEnoughSignatures = HostPatchManagerIntegrityStatus("notEnoughSignatures")
	HostPatchManagerIntegrityStatusValidationError     = HostPatchManagerIntegrityStatus("validationError")
)

func init() {
	t["HostPatchManagerIntegrityStatus"] = reflect.TypeOf((*HostPatchManagerIntegrityStatus)(nil)).Elem()
}

type HostPatchManagerReason string

const (
	HostPatchManagerReasonObsoleted         = HostPatchManagerReason("obsoleted")
	HostPatchManagerReasonMissingPatch      = HostPatchManagerReason("missingPatch")
	HostPatchManagerReasonMissingLib        = HostPatchManagerReason("missingLib")
	HostPatchManagerReasonHasDependentPatch = HostPatchManagerReason("hasDependentPatch")
	HostPatchManagerReasonConflictPatch     = HostPatchManagerReason("conflictPatch")
	HostPatchManagerReasonConflictLib       = HostPatchManagerReason("conflictLib")
)

func init() {
	t["HostPatchManagerReason"] = reflect.TypeOf((*HostPatchManagerReason)(nil)).Elem()
}

type HostPowerOperationType string

const (
	HostPowerOperationTypePowerOn  = HostPowerOperationType("powerOn")
	HostPowerOperationTypePowerOff = HostPowerOperationType("powerOff")
)

func init() {
	t["HostPowerOperationType"] = reflect.TypeOf((*HostPowerOperationType)(nil)).Elem()
}

type HostProfileManagerAnswerFileStatus string

const (
	HostProfileManagerAnswerFileStatusValid   = HostProfileManagerAnswerFileStatus("valid")
	HostProfileManagerAnswerFileStatusInvalid = HostProfileManagerAnswerFileStatus("invalid")
	HostProfileManagerAnswerFileStatusUnknown = HostProfileManagerAnswerFileStatus("unknown")
)

func init() {
	t["HostProfileManagerAnswerFileStatus"] = reflect.TypeOf((*HostProfileManagerAnswerFileStatus)(nil)).Elem()
}

type HostProfileManagerTaskListRequirement string

const (
	HostProfileManagerTaskListRequirementMaintenanceModeRequired = HostProfileManagerTaskListRequirement("maintenanceModeRequired")
	HostProfileManagerTaskListRequirementRebootRequired          = HostProfileManagerTaskListRequirement("rebootRequired")
)

func init() {
	t["HostProfileManagerTaskListRequirement"] = reflect.TypeOf((*HostProfileManagerTaskListRequirement)(nil)).Elem()
}

type HostProtocolEndpointPEType string

const (
	HostProtocolEndpointPETypeBlock = HostProtocolEndpointPEType("block")
	HostProtocolEndpointPETypeNas   = HostProtocolEndpointPEType("nas")
)

func init() {
	t["HostProtocolEndpointPEType"] = reflect.TypeOf((*HostProtocolEndpointPEType)(nil)).Elem()
}

type HostProtocolEndpointProtocolEndpointType string

const (
	HostProtocolEndpointProtocolEndpointTypeScsi  = HostProtocolEndpointProtocolEndpointType("scsi")
	HostProtocolEndpointProtocolEndpointTypeNfs   = HostProtocolEndpointProtocolEndpointType("nfs")
	HostProtocolEndpointProtocolEndpointTypeNfs4x = HostProtocolEndpointProtocolEndpointType("nfs4x")
)

func init() {
	t["HostProtocolEndpointProtocolEndpointType"] = reflect.TypeOf((*HostProtocolEndpointProtocolEndpointType)(nil)).Elem()
}

type HostReplayUnsupportedReason string

const (
	HostReplayUnsupportedReasonIncompatibleProduct = HostReplayUnsupportedReason("incompatibleProduct")
	HostReplayUnsupportedReasonIncompatibleCpu     = HostReplayUnsupportedReason("incompatibleCpu")
	HostReplayUnsupportedReasonHvDisabled          = HostReplayUnsupportedReason("hvDisabled")
	HostReplayUnsupportedReasonCpuidLimitSet       = HostReplayUnsupportedReason("cpuidLimitSet")
	HostReplayUnsupportedReasonOldBIOS             = HostReplayUnsupportedReason("oldBIOS")
	HostReplayUnsupportedReasonUnknown             = HostReplayUnsupportedReason("unknown")
)

func init() {
	t["HostReplayUnsupportedReason"] = reflect.TypeOf((*HostReplayUnsupportedReason)(nil)).Elem()
}

type HostRuntimeInfoNetStackInstanceRuntimeInfoState string

const (
	HostRuntimeInfoNetStackInstanceRuntimeInfoStateInactive     = HostRuntimeInfoNetStackInstanceRuntimeInfoState("inactive")
	HostRuntimeInfoNetStackInstanceRuntimeInfoStateActive       = HostRuntimeInfoNetStackInstanceRuntimeInfoState("active")
	HostRuntimeInfoNetStackInstanceRuntimeInfoStateDeactivating = HostRuntimeInfoNetStackInstanceRuntimeInfoState("deactivating")
	HostRuntimeInfoNetStackInstanceRuntimeInfoStateActivating   = HostRuntimeInfoNetStackInstanceRuntimeInfoState("activating")
)

func init() {
	t["HostRuntimeInfoNetStackInstanceRuntimeInfoState"] = reflect.TypeOf((*HostRuntimeInfoNetStackInstanceRuntimeInfoState)(nil)).Elem()
}

type HostServicePolicy string

const (
	HostServicePolicyOn        = HostServicePolicy("on")
	HostServicePolicyAutomatic = HostServicePolicy("automatic")
	HostServicePolicyOff       = HostServicePolicy("off")
)

func init() {
	t["HostServicePolicy"] = reflect.TypeOf((*HostServicePolicy)(nil)).Elem()
}

type HostSnmpAgentCapability string

const (
	HostSnmpAgentCapabilityCOMPLETE      = HostSnmpAgentCapability("COMPLETE")
	HostSnmpAgentCapabilityDIAGNOSTICS   = HostSnmpAgentCapability("DIAGNOSTICS")
	HostSnmpAgentCapabilityCONFIGURATION = HostSnmpAgentCapability("CONFIGURATION")
)

func init() {
	t["HostSnmpAgentCapability"] = reflect.TypeOf((*HostSnmpAgentCapability)(nil)).Elem()
}

type HostStandbyMode string

const (
	HostStandbyModeEntering = HostStandbyMode("entering")
	HostStandbyModeExiting  = HostStandbyMode("exiting")
	HostStandbyModeIn       = HostStandbyMode("in")
	HostStandbyModeNone     = HostStandbyMode("none")
)

func init() {
	t["HostStandbyMode"] = reflect.TypeOf((*HostStandbyMode)(nil)).Elem()
}

type HostSystemConnectionState string

const (
	HostSystemConnectionStateConnected     = HostSystemConnectionState("connected")
	HostSystemConnectionStateNotResponding = HostSystemConnectionState("notResponding")
	HostSystemConnectionStateDisconnected  = HostSystemConnectionState("disconnected")
)

func init() {
	t["HostSystemConnectionState"] = reflect.TypeOf((*HostSystemConnectionState)(nil)).Elem()
}

type HostSystemIdentificationInfoIdentifier string

const (
	HostSystemIdentificationInfoIdentifierAssetTag          = HostSystemIdentificationInfoIdentifier("AssetTag")
	HostSystemIdentificationInfoIdentifierServiceTag        = HostSystemIdentificationInfoIdentifier("ServiceTag")
	HostSystemIdentificationInfoIdentifierOemSpecificString = HostSystemIdentificationInfoIdentifier("OemSpecificString")
)

func init() {
	t["HostSystemIdentificationInfoIdentifier"] = reflect.TypeOf((*HostSystemIdentificationInfoIdentifier)(nil)).Elem()
}

type HostSystemPowerState string

const (
	HostSystemPowerStatePoweredOn  = HostSystemPowerState("poweredOn")
	HostSystemPowerStatePoweredOff = HostSystemPowerState("poweredOff")
	HostSystemPowerStateStandBy    = HostSystemPowerState("standBy")
	HostSystemPowerStateUnknown    = HostSystemPowerState("unknown")
)

func init() {
	t["HostSystemPowerState"] = reflect.TypeOf((*HostSystemPowerState)(nil)).Elem()
}

type HostUnresolvedVmfsExtentUnresolvedReason string

const (
	HostUnresolvedVmfsExtentUnresolvedReasonDiskIdMismatch = HostUnresolvedVmfsExtentUnresolvedReason("diskIdMismatch")
	HostUnresolvedVmfsExtentUnresolvedReasonUuidConflict   = HostUnresolvedVmfsExtentUnresolvedReason("uuidConflict")
)

func init() {
	t["HostUnresolvedVmfsExtentUnresolvedReason"] = reflect.TypeOf((*HostUnresolvedVmfsExtentUnresolvedReason)(nil)).Elem()
}

type HostUnresolvedVmfsResolutionSpecVmfsUuidResolution string

const (
	HostUnresolvedVmfsResolutionSpecVmfsUuidResolutionResignature = HostUnresolvedVmfsResolutionSpecVmfsUuidResolution("resignature")
	HostUnresolvedVmfsResolutionSpecVmfsUuidResolutionForceMount  = HostUnresolvedVmfsResolutionSpecVmfsUuidResolution("forceMount")
)

func init() {
	t["HostUnresolvedVmfsResolutionSpecVmfsUuidResolution"] = reflect.TypeOf((*HostUnresolvedVmfsResolutionSpecVmfsUuidResolution)(nil)).Elem()
}

type HostVirtualNicManagerNicType string

const (
	HostVirtualNicManagerNicTypeVmotion               = HostVirtualNicManagerNicType("vmotion")
	HostVirtualNicManagerNicTypeFaultToleranceLogging = HostVirtualNicManagerNicType("faultToleranceLogging")
	HostVirtualNicManagerNicTypeVSphereReplication    = HostVirtualNicManagerNicType("vSphereReplication")
	HostVirtualNicManagerNicTypeVSphereReplicationNFC = HostVirtualNicManagerNicType("vSphereReplicationNFC")
	HostVirtualNicManagerNicTypeManagement            = HostVirtualNicManagerNicType("management")
	HostVirtualNicManagerNicTypeVsan                  = HostVirtualNicManagerNicType("vsan")
	HostVirtualNicManagerNicTypeVSphereProvisioning   = HostVirtualNicManagerNicType("vSphereProvisioning")
	HostVirtualNicManagerNicTypeVsanWitness           = HostVirtualNicManagerNicType("vsanWitness")
)

func init() {
	t["HostVirtualNicManagerNicType"] = reflect.TypeOf((*HostVirtualNicManagerNicType)(nil)).Elem()
}

type HostVmciAccessManagerMode string

const (
	HostVmciAccessManagerModeGrant   = HostVmciAccessManagerMode("grant")
	HostVmciAccessManagerModeReplace = HostVmciAccessManagerMode("replace")
	HostVmciAccessManagerModeRevoke  = HostVmciAccessManagerMode("revoke")
)

func init() {
	t["HostVmciAccessManagerMode"] = reflect.TypeOf((*HostVmciAccessManagerMode)(nil)).Elem()
}

type HostVmfsVolumeUnmapPriority string

const (
	HostVmfsVolumeUnmapPriorityNone = HostVmfsVolumeUnmapPriority("none")
	HostVmfsVolumeUnmapPriorityLow  = HostVmfsVolumeUnmapPriority("low")
)

func init() {
	t["HostVmfsVolumeUnmapPriority"] = reflect.TypeOf((*HostVmfsVolumeUnmapPriority)(nil)).Elem()
}

type HttpNfcLeaseState string

const (
	HttpNfcLeaseStateInitializing = HttpNfcLeaseState("initializing")
	HttpNfcLeaseStateReady        = HttpNfcLeaseState("ready")
	HttpNfcLeaseStateDone         = HttpNfcLeaseState("done")
	HttpNfcLeaseStateError        = HttpNfcLeaseState("error")
)

func init() {
	t["HttpNfcLeaseState"] = reflect.TypeOf((*HttpNfcLeaseState)(nil)).Elem()
}

type IncompatibleHostForVmReplicationIncompatibleReason string

const (
	IncompatibleHostForVmReplicationIncompatibleReasonRpo            = IncompatibleHostForVmReplicationIncompatibleReason("rpo")
	IncompatibleHostForVmReplicationIncompatibleReasonNetCompression = IncompatibleHostForVmReplicationIncompatibleReason("netCompression")
)

func init() {
	t["IncompatibleHostForVmReplicationIncompatibleReason"] = reflect.TypeOf((*IncompatibleHostForVmReplicationIncompatibleReason)(nil)).Elem()
}

type InternetScsiSnsDiscoveryMethod string

const (
	InternetScsiSnsDiscoveryMethodIsnsStatic = InternetScsiSnsDiscoveryMethod("isnsStatic")
	InternetScsiSnsDiscoveryMethodIsnsDhcp   = InternetScsiSnsDiscoveryMethod("isnsDhcp")
	InternetScsiSnsDiscoveryMethodIsnsSlp    = InternetScsiSnsDiscoveryMethod("isnsSlp")
)

func init() {
	t["InternetScsiSnsDiscoveryMethod"] = reflect.TypeOf((*InternetScsiSnsDiscoveryMethod)(nil)).Elem()
}

type InvalidDasConfigArgumentEntryForInvalidArgument string

const (
	InvalidDasConfigArgumentEntryForInvalidArgumentAdmissionControl = InvalidDasConfigArgumentEntryForInvalidArgument("admissionControl")
	InvalidDasConfigArgumentEntryForInvalidArgumentUserHeartbeatDs  = InvalidDasConfigArgumentEntryForInvalidArgument("userHeartbeatDs")
	InvalidDasConfigArgumentEntryForInvalidArgumentVmConfig         = InvalidDasConfigArgumentEntryForInvalidArgument("vmConfig")
)

func init() {
	t["InvalidDasConfigArgumentEntryForInvalidArgument"] = reflect.TypeOf((*InvalidDasConfigArgumentEntryForInvalidArgument)(nil)).Elem()
}

type InvalidProfileReferenceHostReason string

const (
	InvalidProfileReferenceHostReasonIncompatibleVersion  = InvalidProfileReferenceHostReason("incompatibleVersion")
	InvalidProfileReferenceHostReasonMissingReferenceHost = InvalidProfileReferenceHostReason("missingReferenceHost")
)

func init() {
	t["InvalidProfileReferenceHostReason"] = reflect.TypeOf((*InvalidProfileReferenceHostReason)(nil)).Elem()
}

type IoFilterOperation string

const (
	IoFilterOperationInstall   = IoFilterOperation("install")
	IoFilterOperationUninstall = IoFilterOperation("uninstall")
	IoFilterOperationUpgrade   = IoFilterOperation("upgrade")
)

func init() {
	t["IoFilterOperation"] = reflect.TypeOf((*IoFilterOperation)(nil)).Elem()
}

type IoFilterType string

const (
	IoFilterTypeCache              = IoFilterType("cache")
	IoFilterTypeReplication        = IoFilterType("replication")
	IoFilterTypeEncryption         = IoFilterType("encryption")
	IoFilterTypeCompression        = IoFilterType("compression")
	IoFilterTypeInspection         = IoFilterType("inspection")
	IoFilterTypeDatastoreIoControl = IoFilterType("datastoreIoControl")
	IoFilterTypeDataProvider       = IoFilterType("dataProvider")
)

func init() {
	t["IoFilterType"] = reflect.TypeOf((*IoFilterType)(nil)).Elem()
}

type IscsiPortInfoPathStatus string

const (
	IscsiPortInfoPathStatusNotUsed    = IscsiPortInfoPathStatus("notUsed")
	IscsiPortInfoPathStatusActive     = IscsiPortInfoPathStatus("active")
	IscsiPortInfoPathStatusStandBy    = IscsiPortInfoPathStatus("standBy")
	IscsiPortInfoPathStatusLastActive = IscsiPortInfoPathStatus("lastActive")
)

func init() {
	t["IscsiPortInfoPathStatus"] = reflect.TypeOf((*IscsiPortInfoPathStatus)(nil)).Elem()
}

type LatencySensitivitySensitivityLevel string

const (
	LatencySensitivitySensitivityLevelLow    = LatencySensitivitySensitivityLevel("low")
	LatencySensitivitySensitivityLevelNormal = LatencySensitivitySensitivityLevel("normal")
	LatencySensitivitySensitivityLevelMedium = LatencySensitivitySensitivityLevel("medium")
	LatencySensitivitySensitivityLevelHigh   = LatencySensitivitySensitivityLevel("high")
	LatencySensitivitySensitivityLevelCustom = LatencySensitivitySensitivityLevel("custom")
)

func init() {
	t["LatencySensitivitySensitivityLevel"] = reflect.TypeOf((*LatencySensitivitySensitivityLevel)(nil)).Elem()
}

type LicenseAssignmentFailedReason string

const (
	LicenseAssignmentFailedReasonKeyEntityMismatch                                    = LicenseAssignmentFailedReason("keyEntityMismatch")
	LicenseAssignmentFailedReasonDowngradeDisallowed                                  = LicenseAssignmentFailedReason("downgradeDisallowed")
	LicenseAssignmentFailedReasonInventoryNotManageableByVirtualCenter                = LicenseAssignmentFailedReason("inventoryNotManageableByVirtualCenter")
	LicenseAssignmentFailedReasonHostsUnmanageableByVirtualCenterWithoutLicenseServer = LicenseAssignmentFailedReason("hostsUnmanageableByVirtualCenterWithoutLicenseServer")
)

func init() {
	t["LicenseAssignmentFailedReason"] = reflect.TypeOf((*LicenseAssignmentFailedReason)(nil)).Elem()
}

type LicenseFeatureInfoSourceRestriction string

const (
	LicenseFeatureInfoSourceRestrictionUnrestricted = LicenseFeatureInfoSourceRestriction("unrestricted")
	LicenseFeatureInfoSourceRestrictionServed       = LicenseFeatureInfoSourceRestriction("served")
	LicenseFeatureInfoSourceRestrictionFile         = LicenseFeatureInfoSourceRestriction("file")
)

func init() {
	t["LicenseFeatureInfoSourceRestriction"] = reflect.TypeOf((*LicenseFeatureInfoSourceRestriction)(nil)).Elem()
}

type LicenseFeatureInfoState string

const (
	LicenseFeatureInfoStateEnabled  = LicenseFeatureInfoState("enabled")
	LicenseFeatureInfoStateDisabled = LicenseFeatureInfoState("disabled")
	LicenseFeatureInfoStateOptional = LicenseFeatureInfoState("optional")
)

func init() {
	t["LicenseFeatureInfoState"] = reflect.TypeOf((*LicenseFeatureInfoState)(nil)).Elem()
}

type LicenseFeatureInfoUnit string

const (
	LicenseFeatureInfoUnitHost       = LicenseFeatureInfoUnit("host")
	LicenseFeatureInfoUnitCpuCore    = LicenseFeatureInfoUnit("cpuCore")
	LicenseFeatureInfoUnitCpuPackage = LicenseFeatureInfoUnit("cpuPackage")
	LicenseFeatureInfoUnitServer     = LicenseFeatureInfoUnit("server")
	LicenseFeatureInfoUnitVm         = LicenseFeatureInfoUnit("vm")
)

func init() {
	t["LicenseFeatureInfoUnit"] = reflect.TypeOf((*LicenseFeatureInfoUnit)(nil)).Elem()
}

type LicenseManagerLicenseKey string

const (
	LicenseManagerLicenseKeyEsxFull    = LicenseManagerLicenseKey("esxFull")
	LicenseManagerLicenseKeyEsxVmtn    = LicenseManagerLicenseKey("esxVmtn")
	LicenseManagerLicenseKeyEsxExpress = LicenseManagerLicenseKey("esxExpress")
	LicenseManagerLicenseKeySan        = LicenseManagerLicenseKey("san")
	LicenseManagerLicenseKeyIscsi      = LicenseManagerLicenseKey("iscsi")
	LicenseManagerLicenseKeyNas        = LicenseManagerLicenseKey("nas")
	LicenseManagerLicenseKeyVsmp       = LicenseManagerLicenseKey("vsmp")
	LicenseManagerLicenseKeyBackup     = LicenseManagerLicenseKey("backup")
	LicenseManagerLicenseKeyVc         = LicenseManagerLicenseKey("vc")
	LicenseManagerLicenseKeyVcExpress  = LicenseManagerLicenseKey("vcExpress")
	LicenseManagerLicenseKeyEsxHost    = LicenseManagerLicenseKey("esxHost")
	LicenseManagerLicenseKeyGsxHost    = LicenseManagerLicenseKey("gsxHost")
	LicenseManagerLicenseKeyServerHost = LicenseManagerLicenseKey("serverHost")
	LicenseManagerLicenseKeyDrsPower   = LicenseManagerLicenseKey("drsPower")
	LicenseManagerLicenseKeyVmotion    = LicenseManagerLicenseKey("vmotion")
	LicenseManagerLicenseKeyDrs        = LicenseManagerLicenseKey("drs")
	LicenseManagerLicenseKeyDas        = LicenseManagerLicenseKey("das")
)

func init() {
	t["LicenseManagerLicenseKey"] = reflect.TypeOf((*LicenseManagerLicenseKey)(nil)).Elem()
}

type LicenseManagerState string

const (
	LicenseManagerStateInitializing = LicenseManagerState("initializing")
	LicenseManagerStateNormal       = LicenseManagerState("normal")
	LicenseManagerStateMarginal     = LicenseManagerState("marginal")
	LicenseManagerStateFault        = LicenseManagerState("fault")
)

func init() {
	t["LicenseManagerState"] = reflect.TypeOf((*LicenseManagerState)(nil)).Elem()
}

type LicenseReservationInfoState string

const (
	LicenseReservationInfoStateNotUsed       = LicenseReservationInfoState("notUsed")
	LicenseReservationInfoStateNoLicense     = LicenseReservationInfoState("noLicense")
	LicenseReservationInfoStateUnlicensedUse = LicenseReservationInfoState("unlicensedUse")
	LicenseReservationInfoStateLicensed      = LicenseReservationInfoState("licensed")
)

func init() {
	t["LicenseReservationInfoState"] = reflect.TypeOf((*LicenseReservationInfoState)(nil)).Elem()
}

type LinkDiscoveryProtocolConfigOperationType string

const (
	LinkDiscoveryProtocolConfigOperationTypeNone      = LinkDiscoveryProtocolConfigOperationType("none")
	LinkDiscoveryProtocolConfigOperationTypeListen    = LinkDiscoveryProtocolConfigOperationType("listen")
	LinkDiscoveryProtocolConfigOperationTypeAdvertise = LinkDiscoveryProtocolConfigOperationType("advertise")
	LinkDiscoveryProtocolConfigOperationTypeBoth      = LinkDiscoveryProtocolConfigOperationType("both")
)

func init() {
	t["LinkDiscoveryProtocolConfigOperationType"] = reflect.TypeOf((*LinkDiscoveryProtocolConfigOperationType)(nil)).Elem()
}

type LinkDiscoveryProtocolConfigProtocolType string

const (
	LinkDiscoveryProtocolConfigProtocolTypeCdp  = LinkDiscoveryProtocolConfigProtocolType("cdp")
	LinkDiscoveryProtocolConfigProtocolTypeLldp = LinkDiscoveryProtocolConfigProtocolType("lldp")
)

func init() {
	t["LinkDiscoveryProtocolConfigProtocolType"] = reflect.TypeOf((*LinkDiscoveryProtocolConfigProtocolType)(nil)).Elem()
}

type ManagedEntityStatus string

const (
	ManagedEntityStatusGray   = ManagedEntityStatus("gray")
	ManagedEntityStatusGreen  = ManagedEntityStatus("green")
	ManagedEntityStatusYellow = ManagedEntityStatus("yellow")
	ManagedEntityStatusRed    = ManagedEntityStatus("red")
)

func init() {
	t["ManagedEntityStatus"] = reflect.TypeOf((*ManagedEntityStatus)(nil)).Elem()
}

type MetricAlarmOperator string

const (
	MetricAlarmOperatorIsAbove = MetricAlarmOperator("isAbove")
	MetricAlarmOperatorIsBelow = MetricAlarmOperator("isBelow")
)

func init() {
	t["MetricAlarmOperator"] = reflect.TypeOf((*MetricAlarmOperator)(nil)).Elem()
}

type MultipathState string

const (
	MultipathStateStandby  = MultipathState("standby")
	MultipathStateActive   = MultipathState("active")
	MultipathStateDisabled = MultipathState("disabled")
	MultipathStateDead     = MultipathState("dead")
	MultipathStateUnknown  = MultipathState("unknown")
)

func init() {
	t["MultipathState"] = reflect.TypeOf((*MultipathState)(nil)).Elem()
}

type NetBIOSConfigInfoMode string

const (
	NetBIOSConfigInfoModeUnknown        = NetBIOSConfigInfoMode("unknown")
	NetBIOSConfigInfoModeEnabled        = NetBIOSConfigInfoMode("enabled")
	NetBIOSConfigInfoModeDisabled       = NetBIOSConfigInfoMode("disabled")
	NetBIOSConfigInfoModeEnabledViaDHCP = NetBIOSConfigInfoMode("enabledViaDHCP")
)

func init() {
	t["NetBIOSConfigInfoMode"] = reflect.TypeOf((*NetBIOSConfigInfoMode)(nil)).Elem()
}

type NetIpConfigInfoIpAddressOrigin string

const (
	NetIpConfigInfoIpAddressOriginOther     = NetIpConfigInfoIpAddressOrigin("other")
	NetIpConfigInfoIpAddressOriginManual    = NetIpConfigInfoIpAddressOrigin("manual")
	NetIpConfigInfoIpAddressOriginDhcp      = NetIpConfigInfoIpAddressOrigin("dhcp")
	NetIpConfigInfoIpAddressOriginLinklayer = NetIpConfigInfoIpAddressOrigin("linklayer")
	NetIpConfigInfoIpAddressOriginRandom    = NetIpConfigInfoIpAddressOrigin("random")
)

func init() {
	t["NetIpConfigInfoIpAddressOrigin"] = reflect.TypeOf((*NetIpConfigInfoIpAddressOrigin)(nil)).Elem()
}

type NetIpConfigInfoIpAddressStatus string

const (
	NetIpConfigInfoIpAddressStatusPreferred    = NetIpConfigInfoIpAddressStatus("preferred")
	NetIpConfigInfoIpAddressStatusDeprecated   = NetIpConfigInfoIpAddressStatus("deprecated")
	NetIpConfigInfoIpAddressStatusInvalid      = NetIpConfigInfoIpAddressStatus("invalid")
	NetIpConfigInfoIpAddressStatusInaccessible = NetIpConfigInfoIpAddressStatus("inaccessible")
	NetIpConfigInfoIpAddressStatusUnknown      = NetIpConfigInfoIpAddressStatus("unknown")
	NetIpConfigInfoIpAddressStatusTentative    = NetIpConfigInfoIpAddressStatus("tentative")
	NetIpConfigInfoIpAddressStatusDuplicate    = NetIpConfigInfoIpAddressStatus("duplicate")
)

func init() {
	t["NetIpConfigInfoIpAddressStatus"] = reflect.TypeOf((*NetIpConfigInfoIpAddressStatus)(nil)).Elem()
}

type NetIpStackInfoEntryType string

const (
	NetIpStackInfoEntryTypeOther   = NetIpStackInfoEntryType("other")
	NetIpStackInfoEntryTypeInvalid = NetIpStackInfoEntryType("invalid")
	NetIpStackInfoEntryTypeDynamic = NetIpStackInfoEntryType("dynamic")
	NetIpStackInfoEntryTypeManual  = NetIpStackInfoEntryType("manual")
)

func init() {
	t["NetIpStackInfoEntryType"] = reflect.TypeOf((*NetIpStackInfoEntryType)(nil)).Elem()
}

type NetIpStackInfoPreference string

const (
	NetIpStackInfoPreferenceReserved = NetIpStackInfoPreference("reserved")
	NetIpStackInfoPreferenceLow      = NetIpStackInfoPreference("low")
	NetIpStackInfoPreferenceMedium   = NetIpStackInfoPreference("medium")
	NetIpStackInfoPreferenceHigh     = NetIpStackInfoPreference("high")
)

func init() {
	t["NetIpStackInfoPreference"] = reflect.TypeOf((*NetIpStackInfoPreference)(nil)).Elem()
}

type NotSupportedDeviceForFTDeviceType string

const (
	NotSupportedDeviceForFTDeviceTypeVirtualVmxnet3            = NotSupportedDeviceForFTDeviceType("virtualVmxnet3")
	NotSupportedDeviceForFTDeviceTypeParaVirtualSCSIController = NotSupportedDeviceForFTDeviceType("paraVirtualSCSIController")
)

func init() {
	t["NotSupportedDeviceForFTDeviceType"] = reflect.TypeOf((*NotSupportedDeviceForFTDeviceType)(nil)).Elem()
}

type NumVirtualCpusIncompatibleReason string

const (
	NumVirtualCpusIncompatibleReasonRecordReplay   = NumVirtualCpusIncompatibleReason("recordReplay")
	NumVirtualCpusIncompatibleReasonFaultTolerance = NumVirtualCpusIncompatibleReason("faultTolerance")
)

func init() {
	t["NumVirtualCpusIncompatibleReason"] = reflect.TypeOf((*NumVirtualCpusIncompatibleReason)(nil)).Elem()
}

type ObjectUpdateKind string

const (
	ObjectUpdateKindModify = ObjectUpdateKind("modify")
	ObjectUpdateKindEnter  = ObjectUpdateKind("enter")
	ObjectUpdateKindLeave  = ObjectUpdateKind("leave")
)

func init() {
	t["ObjectUpdateKind"] = reflect.TypeOf((*ObjectUpdateKind)(nil)).Elem()
}

type OvfConsumerOstNodeType string

const (
	OvfConsumerOstNodeTypeEnvelope                = OvfConsumerOstNodeType("envelope")
	OvfConsumerOstNodeTypeVirtualSystem           = OvfConsumerOstNodeType("virtualSystem")
	OvfConsumerOstNodeTypeVirtualSystemCollection = OvfConsumerOstNodeType("virtualSystemCollection")
)

func init() {
	t["OvfConsumerOstNodeType"] = reflect.TypeOf((*OvfConsumerOstNodeType)(nil)).Elem()
}

type OvfCreateImportSpecParamsDiskProvisioningType string

const (
	OvfCreateImportSpecParamsDiskProvisioningTypeMonolithicSparse     = OvfCreateImportSpecParamsDiskProvisioningType("monolithicSparse")
	OvfCreateImportSpecParamsDiskProvisioningTypeMonolithicFlat       = OvfCreateImportSpecParamsDiskProvisioningType("monolithicFlat")
	OvfCreateImportSpecParamsDiskProvisioningTypeTwoGbMaxExtentSparse = OvfCreateImportSpecParamsDiskProvisioningType("twoGbMaxExtentSparse")
	OvfCreateImportSpecParamsDiskProvisioningTypeTwoGbMaxExtentFlat   = OvfCreateImportSpecParamsDiskProvisioningType("twoGbMaxExtentFlat")
	OvfCreateImportSpecParamsDiskProvisioningTypeThin                 = OvfCreateImportSpecParamsDiskProvisioningType("thin")
	OvfCreateImportSpecParamsDiskProvisioningTypeThick                = OvfCreateImportSpecParamsDiskProvisioningType("thick")
	OvfCreateImportSpecParamsDiskProvisioningTypeSeSparse             = OvfCreateImportSpecParamsDiskProvisioningType("seSparse")
	OvfCreateImportSpecParamsDiskProvisioningTypeEagerZeroedThick     = OvfCreateImportSpecParamsDiskProvisioningType("eagerZeroedThick")
	OvfCreateImportSpecParamsDiskProvisioningTypeSparse               = OvfCreateImportSpecParamsDiskProvisioningType("sparse")
	OvfCreateImportSpecParamsDiskProvisioningTypeFlat                 = OvfCreateImportSpecParamsDiskProvisioningType("flat")
)

func init() {
	t["OvfCreateImportSpecParamsDiskProvisioningType"] = reflect.TypeOf((*OvfCreateImportSpecParamsDiskProvisioningType)(nil)).Elem()
}

type PerfFormat string

const (
	PerfFormatNormal = PerfFormat("normal")
	PerfFormatCsv    = PerfFormat("csv")
)

func init() {
	t["PerfFormat"] = reflect.TypeOf((*PerfFormat)(nil)).Elem()
}

type PerfStatsType string

const (
	PerfStatsTypeAbsolute = PerfStatsType("absolute")
	PerfStatsTypeDelta    = PerfStatsType("delta")
	PerfStatsTypeRate     = PerfStatsType("rate")
)

func init() {
	t["PerfStatsType"] = reflect.TypeOf((*PerfStatsType)(nil)).Elem()
}

type PerfSummaryType string

const (
	PerfSummaryTypeAverage   = PerfSummaryType("average")
	PerfSummaryTypeMaximum   = PerfSummaryType("maximum")
	PerfSummaryTypeMinimum   = PerfSummaryType("minimum")
	PerfSummaryTypeLatest    = PerfSummaryType("latest")
	PerfSummaryTypeSummation = PerfSummaryType("summation")
	PerfSummaryTypeNone      = PerfSummaryType("none")
)

func init() {
	t["PerfSummaryType"] = reflect.TypeOf((*PerfSummaryType)(nil)).Elem()
}

type PerformanceManagerUnit string

const (
	PerformanceManagerUnitPercent            = PerformanceManagerUnit("percent")
	PerformanceManagerUnitKiloBytes          = PerformanceManagerUnit("kiloBytes")
	PerformanceManagerUnitMegaBytes          = PerformanceManagerUnit("megaBytes")
	PerformanceManagerUnitMegaHertz          = PerformanceManagerUnit("megaHertz")
	PerformanceManagerUnitNumber             = PerformanceManagerUnit("number")
	PerformanceManagerUnitMicrosecond        = PerformanceManagerUnit("microsecond")
	PerformanceManagerUnitMillisecond        = PerformanceManagerUnit("millisecond")
	PerformanceManagerUnitSecond             = PerformanceManagerUnit("second")
	PerformanceManagerUnitKiloBytesPerSecond = PerformanceManagerUnit("kiloBytesPerSecond")
	PerformanceManagerUnitMegaBytesPerSecond = PerformanceManagerUnit("megaBytesPerSecond")
	PerformanceManagerUnitWatt               = PerformanceManagerUnit("watt")
	PerformanceManagerUnitJoule              = PerformanceManagerUnit("joule")
	PerformanceManagerUnitTeraBytes          = PerformanceManagerUnit("teraBytes")
)

func init() {
	t["PerformanceManagerUnit"] = reflect.TypeOf((*PerformanceManagerUnit)(nil)).Elem()
}

type PhysicalNicResourcePoolSchedulerDisallowedReason string

const (
	PhysicalNicResourcePoolSchedulerDisallowedReasonUserOptOut          = PhysicalNicResourcePoolSchedulerDisallowedReason("userOptOut")
	PhysicalNicResourcePoolSchedulerDisallowedReasonHardwareUnsupported = PhysicalNicResourcePoolSchedulerDisallowedReason("hardwareUnsupported")
)

func init() {
	t["PhysicalNicResourcePoolSchedulerDisallowedReason"] = reflect.TypeOf((*PhysicalNicResourcePoolSchedulerDisallowedReason)(nil)).Elem()
}

type PhysicalNicVmDirectPathGen2SupportedMode string

const (
	PhysicalNicVmDirectPathGen2SupportedModeUpt = PhysicalNicVmDirectPathGen2SupportedMode("upt")
)

func init() {
	t["PhysicalNicVmDirectPathGen2SupportedMode"] = reflect.TypeOf((*PhysicalNicVmDirectPathGen2SupportedMode)(nil)).Elem()
}

type PlacementAffinityRuleRuleScope string

const (
	PlacementAffinityRuleRuleScopeCluster    = PlacementAffinityRuleRuleScope("cluster")
	PlacementAffinityRuleRuleScopeHost       = PlacementAffinityRuleRuleScope("host")
	PlacementAffinityRuleRuleScopeStoragePod = PlacementAffinityRuleRuleScope("storagePod")
	PlacementAffinityRuleRuleScopeDatastore  = PlacementAffinityRuleRuleScope("datastore")
)

func init() {
	t["PlacementAffinityRuleRuleScope"] = reflect.TypeOf((*PlacementAffinityRuleRuleScope)(nil)).Elem()
}

type PlacementAffinityRuleRuleType string

const (
	PlacementAffinityRuleRuleTypeAffinity         = PlacementAffinityRuleRuleType("affinity")
	PlacementAffinityRuleRuleTypeAntiAffinity     = PlacementAffinityRuleRuleType("antiAffinity")
	PlacementAffinityRuleRuleTypeSoftAffinity     = PlacementAffinityRuleRuleType("softAffinity")
	PlacementAffinityRuleRuleTypeSoftAntiAffinity = PlacementAffinityRuleRuleType("softAntiAffinity")
)

func init() {
	t["PlacementAffinityRuleRuleType"] = reflect.TypeOf((*PlacementAffinityRuleRuleType)(nil)).Elem()
}

type PlacementSpecPlacementType string

const (
	PlacementSpecPlacementTypeCreate      = PlacementSpecPlacementType("create")
	PlacementSpecPlacementTypeReconfigure = PlacementSpecPlacementType("reconfigure")
	PlacementSpecPlacementTypeRelocate    = PlacementSpecPlacementType("relocate")
	PlacementSpecPlacementTypeClone       = PlacementSpecPlacementType("clone")
)

func init() {
	t["PlacementSpecPlacementType"] = reflect.TypeOf((*PlacementSpecPlacementType)(nil)).Elem()
}

type PortGroupConnecteeType string

const (
	PortGroupConnecteeTypeVirtualMachine   = PortGroupConnecteeType("virtualMachine")
	PortGroupConnecteeTypeSystemManagement = PortGroupConnecteeType("systemManagement")
	PortGroupConnecteeTypeHost             = PortGroupConnecteeType("host")
	PortGroupConnecteeTypeUnknown          = PortGroupConnecteeType("unknown")
)

func init() {
	t["PortGroupConnecteeType"] = reflect.TypeOf((*PortGroupConnecteeType)(nil)).Elem()
}

type ProfileExecuteResultStatus string

const (
	ProfileExecuteResultStatusSuccess   = ProfileExecuteResultStatus("success")
	ProfileExecuteResultStatusNeedInput = ProfileExecuteResultStatus("needInput")
	ProfileExecuteResultStatusError     = ProfileExecuteResultStatus("error")
)

func init() {
	t["ProfileExecuteResultStatus"] = reflect.TypeOf((*ProfileExecuteResultStatus)(nil)).Elem()
}

type ProfileNumericComparator string

const (
	ProfileNumericComparatorLessThan         = ProfileNumericComparator("lessThan")
	ProfileNumericComparatorLessThanEqual    = ProfileNumericComparator("lessThanEqual")
	ProfileNumericComparatorEqual            = ProfileNumericComparator("equal")
	ProfileNumericComparatorNotEqual         = ProfileNumericComparator("notEqual")
	ProfileNumericComparatorGreaterThanEqual = ProfileNumericComparator("greaterThanEqual")
	ProfileNumericComparatorGreaterThan      = ProfileNumericComparator("greaterThan")
)

func init() {
	t["ProfileNumericComparator"] = reflect.TypeOf((*ProfileNumericComparator)(nil)).Elem()
}

type PropertyChangeOp string

const (
	PropertyChangeOpAdd            = PropertyChangeOp("add")
	PropertyChangeOpRemove         = PropertyChangeOp("remove")
	PropertyChangeOpAssign         = PropertyChangeOp("assign")
	PropertyChangeOpIndirectRemove = PropertyChangeOp("indirectRemove")
)

func init() {
	t["PropertyChangeOp"] = reflect.TypeOf((*PropertyChangeOp)(nil)).Elem()
}

type QuarantineModeFaultFaultType string

const (
	QuarantineModeFaultFaultTypeNoCompatibleNonQuarantinedHost = QuarantineModeFaultFaultType("NoCompatibleNonQuarantinedHost")
	QuarantineModeFaultFaultTypeCorrectionDisallowed           = QuarantineModeFaultFaultType("CorrectionDisallowed")
	QuarantineModeFaultFaultTypeCorrectionImpact               = QuarantineModeFaultFaultType("CorrectionImpact")
)

func init() {
	t["QuarantineModeFaultFaultType"] = reflect.TypeOf((*QuarantineModeFaultFaultType)(nil)).Elem()
}

type QuiesceMode string

const (
	QuiesceModeApplication = QuiesceMode("application")
	QuiesceModeFilesystem  = QuiesceMode("filesystem")
	QuiesceModeNone        = QuiesceMode("none")
)

func init() {
	t["QuiesceMode"] = reflect.TypeOf((*QuiesceMode)(nil)).Elem()
}

type RecommendationReasonCode string

const (
	RecommendationReasonCodeFairnessCpuAvg                  = RecommendationReasonCode("fairnessCpuAvg")
	RecommendationReasonCodeFairnessMemAvg                  = RecommendationReasonCode("fairnessMemAvg")
	RecommendationReasonCodeJointAffin                      = RecommendationReasonCode("jointAffin")
	RecommendationReasonCodeAntiAffin                       = RecommendationReasonCode("antiAffin")
	RecommendationReasonCodeHostMaint                       = RecommendationReasonCode("hostMaint")
	RecommendationReasonCodeEnterStandby                    = RecommendationReasonCode("enterStandby")
	RecommendationReasonCodeReservationCpu                  = RecommendationReasonCode("reservationCpu")
	RecommendationReasonCodeReservationMem                  = RecommendationReasonCode("reservationMem")
	RecommendationReasonCodePowerOnVm                       = RecommendationReasonCode("powerOnVm")
	RecommendationReasonCodePowerSaving                     = RecommendationReasonCode("powerSaving")
	RecommendationReasonCodeIncreaseCapacity                = RecommendationReasonCode("increaseCapacity")
	RecommendationReasonCodeCheckResource                   = RecommendationReasonCode("checkResource")
	RecommendationReasonCodeUnreservedCapacity              = RecommendationReasonCode("unreservedCapacity")
	RecommendationReasonCodeVmHostHardAffinity              = RecommendationReasonCode("vmHostHardAffinity")
	RecommendationReasonCodeVmHostSoftAffinity              = RecommendationReasonCode("vmHostSoftAffinity")
	RecommendationReasonCodeBalanceDatastoreSpaceUsage      = RecommendationReasonCode("balanceDatastoreSpaceUsage")
	RecommendationReasonCodeBalanceDatastoreIOLoad          = RecommendationReasonCode("balanceDatastoreIOLoad")
	RecommendationReasonCodeBalanceDatastoreIOPSReservation = RecommendationReasonCode("balanceDatastoreIOPSReservation")
	RecommendationReasonCodeDatastoreMaint                  = RecommendationReasonCode("datastoreMaint")
	RecommendationReasonCodeVirtualDiskJointAffin           = RecommendationReasonCode("virtualDiskJointAffin")
	RecommendationReasonCodeVirtualDiskAntiAffin            = RecommendationReasonCode("virtualDiskAntiAffin")
	RecommendationReasonCodeDatastoreSpaceOutage            = RecommendationReasonCode("datastoreSpaceOutage")
	RecommendationReasonCodeStoragePlacement                = RecommendationReasonCode("storagePlacement")
	RecommendationReasonCodeIolbDisabledInternal            = RecommendationReasonCode("iolbDisabledInternal")
	RecommendationReasonCodeXvmotionPlacement               = RecommendationReasonCode("xvmotionPlacement")
	RecommendationReasonCodeNetworkBandwidthReservation     = RecommendationReasonCode("networkBandwidthReservation")
	RecommendationReasonCodeHostInDegradation               = RecommendationReasonCode("hostInDegradation")
	RecommendationReasonCodeHostExitDegradation             = RecommendationReasonCode("hostExitDegradation")
	RecommendationReasonCodeMaxVmsConstraint                = RecommendationReasonCode("maxVmsConstraint")
	RecommendationReasonCodeFtConstraints                   = RecommendationReasonCode("ftConstraints")
)

func init() {
	t["RecommendationReasonCode"] = reflect.TypeOf((*RecommendationReasonCode)(nil)).Elem()
}

type RecommendationType string

const (
	RecommendationTypeV1 = RecommendationType("V1")
)

func init() {
	t["RecommendationType"] = reflect.TypeOf((*RecommendationType)(nil)).Elem()
}

type ReplicationDiskConfigFaultReasonForFault string

const (
	ReplicationDiskConfigFaultReasonForFaultDiskNotFound                           = ReplicationDiskConfigFaultReasonForFault("diskNotFound")
	ReplicationDiskConfigFaultReasonForFaultDiskTypeNotSupported                   = ReplicationDiskConfigFaultReasonForFault("diskTypeNotSupported")
	ReplicationDiskConfigFaultReasonForFaultInvalidDiskKey                         = ReplicationDiskConfigFaultReasonForFault("invalidDiskKey")
	ReplicationDiskConfigFaultReasonForFaultInvalidDiskReplicationId               = ReplicationDiskConfigFaultReasonForFault("invalidDiskReplicationId")
	ReplicationDiskConfigFaultReasonForFaultDuplicateDiskReplicationId             = ReplicationDiskConfigFaultReasonForFault("duplicateDiskReplicationId")
	ReplicationDiskConfigFaultReasonForFaultInvalidPersistentFilePath              = ReplicationDiskConfigFaultReasonForFault("invalidPersistentFilePath")
	ReplicationDiskConfigFaultReasonForFaultReconfigureDiskReplicationIdNotAllowed = ReplicationDiskConfigFaultReasonForFault("reconfigureDiskReplicationIdNotAllowed")
)

func init() {
	t["ReplicationDiskConfigFaultReasonForFault"] = reflect.TypeOf((*ReplicationDiskConfigFaultReasonForFault)(nil)).Elem()
}

type ReplicationVmConfigFaultReasonForFault string

const (
	ReplicationVmConfigFaultReasonForFaultIncompatibleHwVersion                    = ReplicationVmConfigFaultReasonForFault("incompatibleHwVersion")
	ReplicationVmConfigFaultReasonForFaultInvalidVmReplicationId                   = ReplicationVmConfigFaultReasonForFault("invalidVmReplicationId")
	ReplicationVmConfigFaultReasonForFaultInvalidGenerationNumber                  = ReplicationVmConfigFaultReasonForFault("invalidGenerationNumber")
	ReplicationVmConfigFaultReasonForFaultOutOfBoundsRpoValue                      = ReplicationVmConfigFaultReasonForFault("outOfBoundsRpoValue")
	ReplicationVmConfigFaultReasonForFaultInvalidDestinationIpAddress              = ReplicationVmConfigFaultReasonForFault("invalidDestinationIpAddress")
	ReplicationVmConfigFaultReasonForFaultInvalidDestinationPort                   = ReplicationVmConfigFaultReasonForFault("invalidDestinationPort")
	ReplicationVmConfigFaultReasonForFaultInvalidExtraVmOptions                    = ReplicationVmConfigFaultReasonForFault("invalidExtraVmOptions")
	ReplicationVmConfigFaultReasonForFaultStaleGenerationNumber                    = ReplicationVmConfigFaultReasonForFault("staleGenerationNumber")
	ReplicationVmConfigFaultReasonForFaultReconfigureVmReplicationIdNotAllowed     = ReplicationVmConfigFaultReasonForFault("reconfigureVmReplicationIdNotAllowed")
	ReplicationVmConfigFaultReasonForFaultCannotRetrieveVmReplicationConfiguration = ReplicationVmConfigFaultReasonForFault("cannotRetrieveVmReplicationConfiguration")
	ReplicationVmConfigFaultReasonForFaultReplicationAlreadyEnabled                = ReplicationVmConfigFaultReasonForFault("replicationAlreadyEnabled")
	ReplicationVmConfigFaultReasonForFaultInvalidPriorConfiguration                = ReplicationVmConfigFaultReasonForFault("invalidPriorConfiguration")
	ReplicationVmConfigFaultReasonForFaultReplicationNotEnabled                    = ReplicationVmConfigFaultReasonForFault("replicationNotEnabled")
	ReplicationVmConfigFaultReasonForFaultReplicationConfigurationFailed           = ReplicationVmConfigFaultReasonForFault("replicationConfigurationFailed")
	ReplicationVmConfigFaultReasonForFaultEncryptedVm                              = ReplicationVmConfigFaultReasonForFault("encryptedVm")
)

func init() {
	t["ReplicationVmConfigFaultReasonForFault"] = reflect.TypeOf((*ReplicationVmConfigFaultReasonForFault)(nil)).Elem()
}

type ReplicationVmFaultReasonForFault string

const (
	ReplicationVmFaultReasonForFaultNotConfigured      = ReplicationVmFaultReasonForFault("notConfigured")
	ReplicationVmFaultReasonForFaultPoweredOff         = ReplicationVmFaultReasonForFault("poweredOff")
	ReplicationVmFaultReasonForFaultSuspended          = ReplicationVmFaultReasonForFault("suspended")
	ReplicationVmFaultReasonForFaultPoweredOn          = ReplicationVmFaultReasonForFault("poweredOn")
	ReplicationVmFaultReasonForFaultOfflineReplicating = ReplicationVmFaultReasonForFault("offlineReplicating")
	ReplicationVmFaultReasonForFaultInvalidState       = ReplicationVmFaultReasonForFault("invalidState")
	ReplicationVmFaultReasonForFaultInvalidInstanceId  = ReplicationVmFaultReasonForFault("invalidInstanceId")
	ReplicationVmFaultReasonForFaultCloseDiskError     = ReplicationVmFaultReasonForFault("closeDiskError")
)

func init() {
	t["ReplicationVmFaultReasonForFault"] = reflect.TypeOf((*ReplicationVmFaultReasonForFault)(nil)).Elem()
}

type ReplicationVmInProgressFaultActivity string

const (
	ReplicationVmInProgressFaultActivityFullSync = ReplicationVmInProgressFaultActivity("fullSync")
	ReplicationVmInProgressFaultActivityDelta    = ReplicationVmInProgressFaultActivity("delta")
)

func init() {
	t["ReplicationVmInProgressFaultActivity"] = reflect.TypeOf((*ReplicationVmInProgressFaultActivity)(nil)).Elem()
}

type ReplicationVmState string

const (
	ReplicationVmStateNone    = ReplicationVmState("none")
	ReplicationVmStatePaused  = ReplicationVmState("paused")
	ReplicationVmStateSyncing = ReplicationVmState("syncing")
	ReplicationVmStateIdle    = ReplicationVmState("idle")
	ReplicationVmStateActive  = ReplicationVmState("active")
	ReplicationVmStateError   = ReplicationVmState("error")
)

func init() {
	t["ReplicationVmState"] = reflect.TypeOf((*ReplicationVmState)(nil)).Elem()
}

type ScheduledHardwareUpgradeInfoHardwareUpgradePolicy string

const (
	ScheduledHardwareUpgradeInfoHardwareUpgradePolicyNever          = ScheduledHardwareUpgradeInfoHardwareUpgradePolicy("never")
	ScheduledHardwareUpgradeInfoHardwareUpgradePolicyOnSoftPowerOff = ScheduledHardwareUpgradeInfoHardwareUpgradePolicy("onSoftPowerOff")
	ScheduledHardwareUpgradeInfoHardwareUpgradePolicyAlways         = ScheduledHardwareUpgradeInfoHardwareUpgradePolicy("always")
)

func init() {
	t["ScheduledHardwareUpgradeInfoHardwareUpgradePolicy"] = reflect.TypeOf((*ScheduledHardwareUpgradeInfoHardwareUpgradePolicy)(nil)).Elem()
}

type ScheduledHardwareUpgradeInfoHardwareUpgradeStatus string

const (
	ScheduledHardwareUpgradeInfoHardwareUpgradeStatusNone    = ScheduledHardwareUpgradeInfoHardwareUpgradeStatus("none")
	ScheduledHardwareUpgradeInfoHardwareUpgradeStatusPending = ScheduledHardwareUpgradeInfoHardwareUpgradeStatus("pending")
	ScheduledHardwareUpgradeInfoHardwareUpgradeStatusSuccess = ScheduledHardwareUpgradeInfoHardwareUpgradeStatus("success")
	ScheduledHardwareUpgradeInfoHardwareUpgradeStatusFailed  = ScheduledHardwareUpgradeInfoHardwareUpgradeStatus("failed")
)

func init() {
	t["ScheduledHardwareUpgradeInfoHardwareUpgradeStatus"] = reflect.TypeOf((*ScheduledHardwareUpgradeInfoHardwareUpgradeStatus)(nil)).Elem()
}

type ScsiDiskType string

const (
	ScsiDiskTypeNative512   = ScsiDiskType("native512")
	ScsiDiskTypeEmulated512 = ScsiDiskType("emulated512")
	ScsiDiskTypeNative4k    = ScsiDiskType("native4k")
	ScsiDiskTypeUnknown     = ScsiDiskType("unknown")
)

func init() {
	t["ScsiDiskType"] = reflect.TypeOf((*ScsiDiskType)(nil)).Elem()
}

type ScsiLunDescriptorQuality string

const (
	ScsiLunDescriptorQualityHighQuality    = ScsiLunDescriptorQuality("highQuality")
	ScsiLunDescriptorQualityMediumQuality  = ScsiLunDescriptorQuality("mediumQuality")
	ScsiLunDescriptorQualityLowQuality     = ScsiLunDescriptorQuality("lowQuality")
	ScsiLunDescriptorQualityUnknownQuality = ScsiLunDescriptorQuality("unknownQuality")
)

func init() {
	t["ScsiLunDescriptorQuality"] = reflect.TypeOf((*ScsiLunDescriptorQuality)(nil)).Elem()
}

type ScsiLunState string

const (
	ScsiLunStateUnknownState      = ScsiLunState("unknownState")
	ScsiLunStateOk                = ScsiLunState("ok")
	ScsiLunStateError             = ScsiLunState("error")
	ScsiLunStateOff               = ScsiLunState("off")
	ScsiLunStateQuiesced          = ScsiLunState("quiesced")
	ScsiLunStateDegraded          = ScsiLunState("degraded")
	ScsiLunStateLostCommunication = ScsiLunState("lostCommunication")
	ScsiLunStateTimeout           = ScsiLunState("timeout")
)

func init() {
	t["ScsiLunState"] = reflect.TypeOf((*ScsiLunState)(nil)).Elem()
}

type ScsiLunType string

const (
	ScsiLunTypeDisk                   = ScsiLunType("disk")
	ScsiLunTypeTape                   = ScsiLunType("tape")
	ScsiLunTypePrinter                = ScsiLunType("printer")
	ScsiLunTypeProcessor              = ScsiLunType("processor")
	ScsiLunTypeWorm                   = ScsiLunType("worm")
	ScsiLunTypeCdrom                  = ScsiLunType("cdrom")
	ScsiLunTypeScanner                = ScsiLunType("scanner")
	ScsiLunTypeOpticalDevice          = ScsiLunType("opticalDevice")
	ScsiLunTypeMediaChanger           = ScsiLunType("mediaChanger")
	ScsiLunTypeCommunications         = ScsiLunType("communications")
	ScsiLunTypeStorageArrayController = ScsiLunType("storageArrayController")
	ScsiLunTypeEnclosure              = ScsiLunType("enclosure")
	ScsiLunTypeUnknown                = ScsiLunType("unknown")
)

func init() {
	t["ScsiLunType"] = reflect.TypeOf((*ScsiLunType)(nil)).Elem()
}

type ScsiLunVStorageSupportStatus string

const (
	ScsiLunVStorageSupportStatusVStorageSupported   = ScsiLunVStorageSupportStatus("vStorageSupported")
	ScsiLunVStorageSupportStatusVStorageUnsupported = ScsiLunVStorageSupportStatus("vStorageUnsupported")
	ScsiLunVStorageSupportStatusVStorageUnknown     = ScsiLunVStorageSupportStatus("vStorageUnknown")
)

func init() {
	t["ScsiLunVStorageSupportStatus"] = reflect.TypeOf((*ScsiLunVStorageSupportStatus)(nil)).Elem()
}

type SessionManagerHttpServiceRequestSpecMethod string

const (
	SessionManagerHttpServiceRequestSpecMethodHttpOptions = SessionManagerHttpServiceRequestSpecMethod("httpOptions")
	SessionManagerHttpServiceRequestSpecMethodHttpGet     = SessionManagerHttpServiceRequestSpecMethod("httpGet")
	SessionManagerHttpServiceRequestSpecMethodHttpHead    = SessionManagerHttpServiceRequestSpecMethod("httpHead")
	SessionManagerHttpServiceRequestSpecMethodHttpPost    = SessionManagerHttpServiceRequestSpecMethod("httpPost")
	SessionManagerHttpServiceRequestSpecMethodHttpPut     = SessionManagerHttpServiceRequestSpecMethod("httpPut")
	SessionManagerHttpServiceRequestSpecMethodHttpDelete  = SessionManagerHttpServiceRequestSpecMethod("httpDelete")
	SessionManagerHttpServiceRequestSpecMethodHttpTrace   = SessionManagerHttpServiceRequestSpecMethod("httpTrace")
	SessionManagerHttpServiceRequestSpecMethodHttpConnect = SessionManagerHttpServiceRequestSpecMethod("httpConnect")
)

func init() {
	t["SessionManagerHttpServiceRequestSpecMethod"] = reflect.TypeOf((*SessionManagerHttpServiceRequestSpecMethod)(nil)).Elem()
}

type SharesLevel string

const (
	SharesLevelLow    = SharesLevel("low")
	SharesLevelNormal = SharesLevel("normal")
	SharesLevelHigh   = SharesLevel("high")
	SharesLevelCustom = SharesLevel("custom")
)

func init() {
	t["SharesLevel"] = reflect.TypeOf((*SharesLevel)(nil)).Elem()
}

type SimpleCommandEncoding string

const (
	SimpleCommandEncodingCSV    = SimpleCommandEncoding("CSV")
	SimpleCommandEncodingHEX    = SimpleCommandEncoding("HEX")
	SimpleCommandEncodingSTRING = SimpleCommandEncoding("STRING")
)

func init() {
	t["SimpleCommandEncoding"] = reflect.TypeOf((*SimpleCommandEncoding)(nil)).Elem()
}

type SlpDiscoveryMethod string

const (
	SlpDiscoveryMethodSlpDhcp          = SlpDiscoveryMethod("slpDhcp")
	SlpDiscoveryMethodSlpAutoUnicast   = SlpDiscoveryMethod("slpAutoUnicast")
	SlpDiscoveryMethodSlpAutoMulticast = SlpDiscoveryMethod("slpAutoMulticast")
	SlpDiscoveryMethodSlpManual        = SlpDiscoveryMethod("slpManual")
)

func init() {
	t["SlpDiscoveryMethod"] = reflect.TypeOf((*SlpDiscoveryMethod)(nil)).Elem()
}

type SoftwarePackageConstraint string

const (
	SoftwarePackageConstraintEquals           = SoftwarePackageConstraint("equals")
	SoftwarePackageConstraintLessThan         = SoftwarePackageConstraint("lessThan")
	SoftwarePackageConstraintLessThanEqual    = SoftwarePackageConstraint("lessThanEqual")
	SoftwarePackageConstraintGreaterThanEqual = SoftwarePackageConstraint("greaterThanEqual")
	SoftwarePackageConstraintGreaterThan      = SoftwarePackageConstraint("greaterThan")
)

func init() {
	t["SoftwarePackageConstraint"] = reflect.TypeOf((*SoftwarePackageConstraint)(nil)).Elem()
}

type SoftwarePackageVibType string

const (
	SoftwarePackageVibTypeBootbank = SoftwarePackageVibType("bootbank")
	SoftwarePackageVibTypeTools    = SoftwarePackageVibType("tools")
	SoftwarePackageVibTypeMeta     = SoftwarePackageVibType("meta")
)

func init() {
	t["SoftwarePackageVibType"] = reflect.TypeOf((*SoftwarePackageVibType)(nil)).Elem()
}

type StateAlarmOperator string

const (
	StateAlarmOperatorIsEqual   = StateAlarmOperator("isEqual")
	StateAlarmOperatorIsUnequal = StateAlarmOperator("isUnequal")
)

func init() {
	t["StateAlarmOperator"] = reflect.TypeOf((*StateAlarmOperator)(nil)).Elem()
}

type StorageDrsPodConfigInfoBehavior string

const (
	StorageDrsPodConfigInfoBehaviorManual    = StorageDrsPodConfigInfoBehavior("manual")
	StorageDrsPodConfigInfoBehaviorAutomated = StorageDrsPodConfigInfoBehavior("automated")
)

func init() {
	t["StorageDrsPodConfigInfoBehavior"] = reflect.TypeOf((*StorageDrsPodConfigInfoBehavior)(nil)).Elem()
}

type StorageDrsSpaceLoadBalanceConfigSpaceThresholdMode string

const (
	StorageDrsSpaceLoadBalanceConfigSpaceThresholdModeUtilization = StorageDrsSpaceLoadBalanceConfigSpaceThresholdMode("utilization")
	StorageDrsSpaceLoadBalanceConfigSpaceThresholdModeFreeSpace   = StorageDrsSpaceLoadBalanceConfigSpaceThresholdMode("freeSpace")
)

func init() {
	t["StorageDrsSpaceLoadBalanceConfigSpaceThresholdMode"] = reflect.TypeOf((*StorageDrsSpaceLoadBalanceConfigSpaceThresholdMode)(nil)).Elem()
}

type StorageIORMThresholdMode string

const (
	StorageIORMThresholdModeAutomatic = StorageIORMThresholdMode("automatic")
	StorageIORMThresholdModeManual    = StorageIORMThresholdMode("manual")
)

func init() {
	t["StorageIORMThresholdMode"] = reflect.TypeOf((*StorageIORMThresholdMode)(nil)).Elem()
}

type StoragePlacementSpecPlacementType string

const (
	StoragePlacementSpecPlacementTypeCreate      = StoragePlacementSpecPlacementType("create")
	StoragePlacementSpecPlacementTypeReconfigure = StoragePlacementSpecPlacementType("reconfigure")
	StoragePlacementSpecPlacementTypeRelocate    = StoragePlacementSpecPlacementType("relocate")
	StoragePlacementSpecPlacementTypeClone       = StoragePlacementSpecPlacementType("clone")
)

func init() {
	t["StoragePlacementSpecPlacementType"] = reflect.TypeOf((*StoragePlacementSpecPlacementType)(nil)).Elem()
}

type TaskFilterSpecRecursionOption string

const (
	TaskFilterSpecRecursionOptionSelf     = TaskFilterSpecRecursionOption("self")
	TaskFilterSpecRecursionOptionChildren = TaskFilterSpecRecursionOption("children")
	TaskFilterSpecRecursionOptionAll      = TaskFilterSpecRecursionOption("all")
)

func init() {
	t["TaskFilterSpecRecursionOption"] = reflect.TypeOf((*TaskFilterSpecRecursionOption)(nil)).Elem()
}

type TaskFilterSpecTimeOption string

const (
	TaskFilterSpecTimeOptionQueuedTime    = TaskFilterSpecTimeOption("queuedTime")
	TaskFilterSpecTimeOptionStartedTime   = TaskFilterSpecTimeOption("startedTime")
	TaskFilterSpecTimeOptionCompletedTime = TaskFilterSpecTimeOption("completedTime")
)

func init() {
	t["TaskFilterSpecTimeOption"] = reflect.TypeOf((*TaskFilterSpecTimeOption)(nil)).Elem()
}

type TaskInfoState string

const (
	TaskInfoStateQueued  = TaskInfoState("queued")
	TaskInfoStateRunning = TaskInfoState("running")
	TaskInfoStateSuccess = TaskInfoState("success")
	TaskInfoStateError   = TaskInfoState("error")
)

func init() {
	t["TaskInfoState"] = reflect.TypeOf((*TaskInfoState)(nil)).Elem()
}

type ThirdPartyLicenseAssignmentFailedReason string

const (
	ThirdPartyLicenseAssignmentFailedReasonLicenseAssignmentFailed = ThirdPartyLicenseAssignmentFailedReason("licenseAssignmentFailed")
	ThirdPartyLicenseAssignmentFailedReasonModuleNotInstalled      = ThirdPartyLicenseAssignmentFailedReason("moduleNotInstalled")
)

func init() {
	t["ThirdPartyLicenseAssignmentFailedReason"] = reflect.TypeOf((*ThirdPartyLicenseAssignmentFailedReason)(nil)).Elem()
}

type UpgradePolicy string

const (
	UpgradePolicyManual              = UpgradePolicy("manual")
	UpgradePolicyUpgradeAtPowerCycle = UpgradePolicy("upgradeAtPowerCycle")
)

func init() {
	t["UpgradePolicy"] = reflect.TypeOf((*UpgradePolicy)(nil)).Elem()
}

type VAppAutoStartAction string

const (
	VAppAutoStartActionNone          = VAppAutoStartAction("none")
	VAppAutoStartActionPowerOn       = VAppAutoStartAction("powerOn")
	VAppAutoStartActionPowerOff      = VAppAutoStartAction("powerOff")
	VAppAutoStartActionGuestShutdown = VAppAutoStartAction("guestShutdown")
	VAppAutoStartActionSuspend       = VAppAutoStartAction("suspend")
)

func init() {
	t["VAppAutoStartAction"] = reflect.TypeOf((*VAppAutoStartAction)(nil)).Elem()
}

type VAppCloneSpecProvisioningType string

const (
	VAppCloneSpecProvisioningTypeSameAsSource = VAppCloneSpecProvisioningType("sameAsSource")
	VAppCloneSpecProvisioningTypeThin         = VAppCloneSpecProvisioningType("thin")
	VAppCloneSpecProvisioningTypeThick        = VAppCloneSpecProvisioningType("thick")
)

func init() {
	t["VAppCloneSpecProvisioningType"] = reflect.TypeOf((*VAppCloneSpecProvisioningType)(nil)).Elem()
}

type VAppIPAssignmentInfoAllocationSchemes string

const (
	VAppIPAssignmentInfoAllocationSchemesDhcp   = VAppIPAssignmentInfoAllocationSchemes("dhcp")
	VAppIPAssignmentInfoAllocationSchemesOvfenv = VAppIPAssignmentInfoAllocationSchemes("ovfenv")
)

func init() {
	t["VAppIPAssignmentInfoAllocationSchemes"] = reflect.TypeOf((*VAppIPAssignmentInfoAllocationSchemes)(nil)).Elem()
}

type VAppIPAssignmentInfoIpAllocationPolicy string

const (
	VAppIPAssignmentInfoIpAllocationPolicyDhcpPolicy           = VAppIPAssignmentInfoIpAllocationPolicy("dhcpPolicy")
	VAppIPAssignmentInfoIpAllocationPolicyTransientPolicy      = VAppIPAssignmentInfoIpAllocationPolicy("transientPolicy")
	VAppIPAssignmentInfoIpAllocationPolicyFixedPolicy          = VAppIPAssignmentInfoIpAllocationPolicy("fixedPolicy")
	VAppIPAssignmentInfoIpAllocationPolicyFixedAllocatedPolicy = VAppIPAssignmentInfoIpAllocationPolicy("fixedAllocatedPolicy")
)

func init() {
	t["VAppIPAssignmentInfoIpAllocationPolicy"] = reflect.TypeOf((*VAppIPAssignmentInfoIpAllocationPolicy)(nil)).Elem()
}

type VAppIPAssignmentInfoProtocols string

const (
	VAppIPAssignmentInfoProtocolsIPv4 = VAppIPAssignmentInfoProtocols("IPv4")
	VAppIPAssignmentInfoProtocolsIPv6 = VAppIPAssignmentInfoProtocols("IPv6")
)

func init() {
	t["VAppIPAssignmentInfoProtocols"] = reflect.TypeOf((*VAppIPAssignmentInfoProtocols)(nil)).Elem()
}

type VFlashModuleNotSupportedReason string

const (
	VFlashModuleNotSupportedReasonCacheModeNotSupported            = VFlashModuleNotSupportedReason("CacheModeNotSupported")
	VFlashModuleNotSupportedReasonCacheConsistencyTypeNotSupported = VFlashModuleNotSupportedReason("CacheConsistencyTypeNotSupported")
	VFlashModuleNotSupportedReasonCacheBlockSizeNotSupported       = VFlashModuleNotSupportedReason("CacheBlockSizeNotSupported")
	VFlashModuleNotSupportedReasonCacheReservationNotSupported     = VFlashModuleNotSupportedReason("CacheReservationNotSupported")
	VFlashModuleNotSupportedReasonDiskSizeNotSupported             = VFlashModuleNotSupportedReason("DiskSizeNotSupported")
)

func init() {
	t["VFlashModuleNotSupportedReason"] = reflect.TypeOf((*VFlashModuleNotSupportedReason)(nil)).Elem()
}

type VMotionCompatibilityType string

const (
	VMotionCompatibilityTypeCpu      = VMotionCompatibilityType("cpu")
	VMotionCompatibilityTypeSoftware = VMotionCompatibilityType("software")
)

func init() {
	t["VMotionCompatibilityType"] = reflect.TypeOf((*VMotionCompatibilityType)(nil)).Elem()
}

type VMwareDVSTeamingMatchStatus string

const (
	VMwareDVSTeamingMatchStatusIphashMatch       = VMwareDVSTeamingMatchStatus("iphashMatch")
	VMwareDVSTeamingMatchStatusNonIphashMatch    = VMwareDVSTeamingMatchStatus("nonIphashMatch")
	VMwareDVSTeamingMatchStatusIphashMismatch    = VMwareDVSTeamingMatchStatus("iphashMismatch")
	VMwareDVSTeamingMatchStatusNonIphashMismatch = VMwareDVSTeamingMatchStatus("nonIphashMismatch")
)

func init() {
	t["VMwareDVSTeamingMatchStatus"] = reflect.TypeOf((*VMwareDVSTeamingMatchStatus)(nil)).Elem()
}

type VMwareDVSVspanSessionEncapType string

const (
	VMwareDVSVspanSessionEncapTypeGre     = VMwareDVSVspanSessionEncapType("gre")
	VMwareDVSVspanSessionEncapTypeErspan2 = VMwareDVSVspanSessionEncapType("erspan2")
	VMwareDVSVspanSessionEncapTypeErspan3 = VMwareDVSVspanSessionEncapType("erspan3")
)

func init() {
	t["VMwareDVSVspanSessionEncapType"] = reflect.TypeOf((*VMwareDVSVspanSessionEncapType)(nil)).Elem()
}

type VMwareDVSVspanSessionType string

const (
	VMwareDVSVspanSessionTypeMixedDestMirror                = VMwareDVSVspanSessionType("mixedDestMirror")
	VMwareDVSVspanSessionTypeDvPortMirror                   = VMwareDVSVspanSessionType("dvPortMirror")
	VMwareDVSVspanSessionTypeRemoteMirrorSource             = VMwareDVSVspanSessionType("remoteMirrorSource")
	VMwareDVSVspanSessionTypeRemoteMirrorDest               = VMwareDVSVspanSessionType("remoteMirrorDest")
	VMwareDVSVspanSessionTypeEncapsulatedRemoteMirrorSource = VMwareDVSVspanSessionType("encapsulatedRemoteMirrorSource")
)

func init() {
	t["VMwareDVSVspanSessionType"] = reflect.TypeOf((*VMwareDVSVspanSessionType)(nil)).Elem()
}

type VMwareDvsLacpApiVersion string

const (
	VMwareDvsLacpApiVersionSingleLag   = VMwareDvsLacpApiVersion("singleLag")
	VMwareDvsLacpApiVersionMultipleLag = VMwareDvsLacpApiVersion("multipleLag")
)

func init() {
	t["VMwareDvsLacpApiVersion"] = reflect.TypeOf((*VMwareDvsLacpApiVersion)(nil)).Elem()
}

type VMwareDvsLacpLoadBalanceAlgorithm string

const (
	VMwareDvsLacpLoadBalanceAlgorithmSrcMac                  = VMwareDvsLacpLoadBalanceAlgorithm("srcMac")
	VMwareDvsLacpLoadBalanceAlgorithmDestMac                 = VMwareDvsLacpLoadBalanceAlgorithm("destMac")
	VMwareDvsLacpLoadBalanceAlgorithmSrcDestMac              = VMwareDvsLacpLoadBalanceAlgorithm("srcDestMac")
	VMwareDvsLacpLoadBalanceAlgorithmDestIpVlan              = VMwareDvsLacpLoadBalanceAlgorithm("destIpVlan")
	VMwareDvsLacpLoadBalanceAlgorithmSrcIpVlan               = VMwareDvsLacpLoadBalanceAlgorithm("srcIpVlan")
	VMwareDvsLacpLoadBalanceAlgorithmSrcDestIpVlan           = VMwareDvsLacpLoadBalanceAlgorithm("srcDestIpVlan")
	VMwareDvsLacpLoadBalanceAlgorithmDestTcpUdpPort          = VMwareDvsLacpLoadBalanceAlgorithm("destTcpUdpPort")
	VMwareDvsLacpLoadBalanceAlgorithmSrcTcpUdpPort           = VMwareDvsLacpLoadBalanceAlgorithm("srcTcpUdpPort")
	VMwareDvsLacpLoadBalanceAlgorithmSrcDestTcpUdpPort       = VMwareDvsLacpLoadBalanceAlgorithm("srcDestTcpUdpPort")
	VMwareDvsLacpLoadBalanceAlgorithmDestIpTcpUdpPort        = VMwareDvsLacpLoadBalanceAlgorithm("destIpTcpUdpPort")
	VMwareDvsLacpLoadBalanceAlgorithmSrcIpTcpUdpPort         = VMwareDvsLacpLoadBalanceAlgorithm("srcIpTcpUdpPort")
	VMwareDvsLacpLoadBalanceAlgorithmSrcDestIpTcpUdpPort     = VMwareDvsLacpLoadBalanceAlgorithm("srcDestIpTcpUdpPort")
	VMwareDvsLacpLoadBalanceAlgorithmDestIpTcpUdpPortVlan    = VMwareDvsLacpLoadBalanceAlgorithm("destIpTcpUdpPortVlan")
	VMwareDvsLacpLoadBalanceAlgorithmSrcIpTcpUdpPortVlan     = VMwareDvsLacpLoadBalanceAlgorithm("srcIpTcpUdpPortVlan")
	VMwareDvsLacpLoadBalanceAlgorithmSrcDestIpTcpUdpPortVlan = VMwareDvsLacpLoadBalanceAlgorithm("srcDestIpTcpUdpPortVlan")
	VMwareDvsLacpLoadBalanceAlgorithmDestIp                  = VMwareDvsLacpLoadBalanceAlgorithm("destIp")
	VMwareDvsLacpLoadBalanceAlgorithmSrcIp                   = VMwareDvsLacpLoadBalanceAlgorithm("srcIp")
	VMwareDvsLacpLoadBalanceAlgorithmSrcDestIp               = VMwareDvsLacpLoadBalanceAlgorithm("srcDestIp")
	VMwareDvsLacpLoadBalanceAlgorithmVlan                    = VMwareDvsLacpLoadBalanceAlgorithm("vlan")
	VMwareDvsLacpLoadBalanceAlgorithmSrcPortId               = VMwareDvsLacpLoadBalanceAlgorithm("srcPortId")
)

func init() {
	t["VMwareDvsLacpLoadBalanceAlgorithm"] = reflect.TypeOf((*VMwareDvsLacpLoadBalanceAlgorithm)(nil)).Elem()
}

type VMwareDvsMulticastFilteringMode string

const (
	VMwareDvsMulticastFilteringModeLegacyFiltering = VMwareDvsMulticastFilteringMode("legacyFiltering")
	VMwareDvsMulticastFilteringModeSnooping        = VMwareDvsMulticastFilteringMode("snooping")
)

func init() {
	t["VMwareDvsMulticastFilteringMode"] = reflect.TypeOf((*VMwareDvsMulticastFilteringMode)(nil)).Elem()
}

type VMwareUplinkLacpMode string

const (
	VMwareUplinkLacpModeActive  = VMwareUplinkLacpMode("active")
	VMwareUplinkLacpModePassive = VMwareUplinkLacpMode("passive")
)

func init() {
	t["VMwareUplinkLacpMode"] = reflect.TypeOf((*VMwareUplinkLacpMode)(nil)).Elem()
}

type VStorageObjectConsumptionType string

const (
	VStorageObjectConsumptionTypeDisk = VStorageObjectConsumptionType("disk")
)

func init() {
	t["VStorageObjectConsumptionType"] = reflect.TypeOf((*VStorageObjectConsumptionType)(nil)).Elem()
}

type ValidateMigrationTestType string

const (
	ValidateMigrationTestTypeSourceTests            = ValidateMigrationTestType("sourceTests")
	ValidateMigrationTestTypeCompatibilityTests     = ValidateMigrationTestType("compatibilityTests")
	ValidateMigrationTestTypeDiskAccessibilityTests = ValidateMigrationTestType("diskAccessibilityTests")
	ValidateMigrationTestTypeResourceTests          = ValidateMigrationTestType("resourceTests")
)

func init() {
	t["ValidateMigrationTestType"] = reflect.TypeOf((*ValidateMigrationTestType)(nil)).Elem()
}

type VchaClusterMode string

const (
	VchaClusterModeEnabled     = VchaClusterMode("enabled")
	VchaClusterModeDisabled    = VchaClusterMode("disabled")
	VchaClusterModeMaintenance = VchaClusterMode("maintenance")
)

func init() {
	t["VchaClusterMode"] = reflect.TypeOf((*VchaClusterMode)(nil)).Elem()
}

type VchaClusterState string

const (
	VchaClusterStateHealthy  = VchaClusterState("healthy")
	VchaClusterStateDegraded = VchaClusterState("degraded")
	VchaClusterStateIsolated = VchaClusterState("isolated")
)

func init() {
	t["VchaClusterState"] = reflect.TypeOf((*VchaClusterState)(nil)).Elem()
}

type VchaNodeRole string

const (
	VchaNodeRoleActive  = VchaNodeRole("active")
	VchaNodeRolePassive = VchaNodeRole("passive")
	VchaNodeRoleWitness = VchaNodeRole("witness")
)

func init() {
	t["VchaNodeRole"] = reflect.TypeOf((*VchaNodeRole)(nil)).Elem()
}

type VchaNodeState string

const (
	VchaNodeStateUp   = VchaNodeState("up")
	VchaNodeStateDown = VchaNodeState("down")
)

func init() {
	t["VchaNodeState"] = reflect.TypeOf((*VchaNodeState)(nil)).Elem()
}

type VchaState string

const (
	VchaStateConfigured    = VchaState("configured")
	VchaStateNotConfigured = VchaState("notConfigured")
	VchaStateInvalid       = VchaState("invalid")
	VchaStatePrepared      = VchaState("prepared")
)

func init() {
	t["VchaState"] = reflect.TypeOf((*VchaState)(nil)).Elem()
}

type VirtualAppVAppState string

const (
	VirtualAppVAppStateStarted  = VirtualAppVAppState("started")
	VirtualAppVAppStateStopped  = VirtualAppVAppState("stopped")
	VirtualAppVAppStateStarting = VirtualAppVAppState("starting")
	VirtualAppVAppStateStopping = VirtualAppVAppState("stopping")
)

func init() {
	t["VirtualAppVAppState"] = reflect.TypeOf((*VirtualAppVAppState)(nil)).Elem()
}

type VirtualDeviceConfigSpecFileOperation string

const (
	VirtualDeviceConfigSpecFileOperationCreate  = VirtualDeviceConfigSpecFileOperation("create")
	VirtualDeviceConfigSpecFileOperationDestroy = VirtualDeviceConfigSpecFileOperation("destroy")
	VirtualDeviceConfigSpecFileOperationReplace = VirtualDeviceConfigSpecFileOperation("replace")
)

func init() {
	t["VirtualDeviceConfigSpecFileOperation"] = reflect.TypeOf((*VirtualDeviceConfigSpecFileOperation)(nil)).Elem()
}

type VirtualDeviceConfigSpecOperation string

const (
	VirtualDeviceConfigSpecOperationAdd    = VirtualDeviceConfigSpecOperation("add")
	VirtualDeviceConfigSpecOperationRemove = VirtualDeviceConfigSpecOperation("remove")
	VirtualDeviceConfigSpecOperationEdit   = VirtualDeviceConfigSpecOperation("edit")
)

func init() {
	t["VirtualDeviceConfigSpecOperation"] = reflect.TypeOf((*VirtualDeviceConfigSpecOperation)(nil)).Elem()
}

type VirtualDeviceConnectInfoStatus string

const (
	VirtualDeviceConnectInfoStatusOk                 = VirtualDeviceConnectInfoStatus("ok")
	VirtualDeviceConnectInfoStatusRecoverableError   = VirtualDeviceConnectInfoStatus("recoverableError")
	VirtualDeviceConnectInfoStatusUnrecoverableError = VirtualDeviceConnectInfoStatus("unrecoverableError")
	VirtualDeviceConnectInfoStatusUntried            = VirtualDeviceConnectInfoStatus("untried")
)

func init() {
	t["VirtualDeviceConnectInfoStatus"] = reflect.TypeOf((*VirtualDeviceConnectInfoStatus)(nil)).Elem()
}

type VirtualDeviceFileExtension string

const (
	VirtualDeviceFileExtensionIso  = VirtualDeviceFileExtension("iso")
	VirtualDeviceFileExtensionFlp  = VirtualDeviceFileExtension("flp")
	VirtualDeviceFileExtensionVmdk = VirtualDeviceFileExtension("vmdk")
	VirtualDeviceFileExtensionDsk  = VirtualDeviceFileExtension("dsk")
	VirtualDeviceFileExtensionRdm  = VirtualDeviceFileExtension("rdm")
)

func init() {
	t["VirtualDeviceFileExtension"] = reflect.TypeOf((*VirtualDeviceFileExtension)(nil)).Elem()
}

type VirtualDeviceURIBackingOptionDirection string

const (
	VirtualDeviceURIBackingOptionDirectionServer = VirtualDeviceURIBackingOptionDirection("server")
	VirtualDeviceURIBackingOptionDirectionClient = VirtualDeviceURIBackingOptionDirection("client")
)

func init() {
	t["VirtualDeviceURIBackingOptionDirection"] = reflect.TypeOf((*VirtualDeviceURIBackingOptionDirection)(nil)).Elem()
}

type VirtualDiskAdapterType string

const (
	VirtualDiskAdapterTypeIde      = VirtualDiskAdapterType("ide")
	VirtualDiskAdapterTypeBusLogic = VirtualDiskAdapterType("busLogic")
	VirtualDiskAdapterTypeLsiLogic = VirtualDiskAdapterType("lsiLogic")
)

func init() {
	t["VirtualDiskAdapterType"] = reflect.TypeOf((*VirtualDiskAdapterType)(nil)).Elem()
}

type VirtualDiskCompatibilityMode string

const (
	VirtualDiskCompatibilityModeVirtualMode  = VirtualDiskCompatibilityMode("virtualMode")
	VirtualDiskCompatibilityModePhysicalMode = VirtualDiskCompatibilityMode("physicalMode")
)

func init() {
	t["VirtualDiskCompatibilityMode"] = reflect.TypeOf((*VirtualDiskCompatibilityMode)(nil)).Elem()
}

type VirtualDiskDeltaDiskFormat string

const (
	VirtualDiskDeltaDiskFormatRedoLogFormat  = VirtualDiskDeltaDiskFormat("redoLogFormat")
	VirtualDiskDeltaDiskFormatNativeFormat   = VirtualDiskDeltaDiskFormat("nativeFormat")
	VirtualDiskDeltaDiskFormatSeSparseFormat = VirtualDiskDeltaDiskFormat("seSparseFormat")
)

func init() {
	t["VirtualDiskDeltaDiskFormat"] = reflect.TypeOf((*VirtualDiskDeltaDiskFormat)(nil)).Elem()
}

type VirtualDiskDeltaDiskFormatVariant string

const (
	VirtualDiskDeltaDiskFormatVariantVmfsSparseVariant = VirtualDiskDeltaDiskFormatVariant("vmfsSparseVariant")
	VirtualDiskDeltaDiskFormatVariantVsanSparseVariant = VirtualDiskDeltaDiskFormatVariant("vsanSparseVariant")
)

func init() {
	t["VirtualDiskDeltaDiskFormatVariant"] = reflect.TypeOf((*VirtualDiskDeltaDiskFormatVariant)(nil)).Elem()
}

type VirtualDiskMode string

const (
	VirtualDiskModePersistent                = VirtualDiskMode("persistent")
	VirtualDiskModeNonpersistent             = VirtualDiskMode("nonpersistent")
	VirtualDiskModeUndoable                  = VirtualDiskMode("undoable")
	VirtualDiskModeIndependent_persistent    = VirtualDiskMode("independent_persistent")
	VirtualDiskModeIndependent_nonpersistent = VirtualDiskMode("independent_nonpersistent")
	VirtualDiskModeAppend                    = VirtualDiskMode("append")
)

func init() {
	t["VirtualDiskMode"] = reflect.TypeOf((*VirtualDiskMode)(nil)).Elem()
}

type VirtualDiskSharing string

const (
	VirtualDiskSharingSharingNone        = VirtualDiskSharing("sharingNone")
	VirtualDiskSharingSharingMultiWriter = VirtualDiskSharing("sharingMultiWriter")
)

func init() {
	t["VirtualDiskSharing"] = reflect.TypeOf((*VirtualDiskSharing)(nil)).Elem()
}

type VirtualDiskType string

const (
	VirtualDiskTypePreallocated     = VirtualDiskType("preallocated")
	VirtualDiskTypeThin             = VirtualDiskType("thin")
	VirtualDiskTypeSeSparse         = VirtualDiskType("seSparse")
	VirtualDiskTypeRdm              = VirtualDiskType("rdm")
	VirtualDiskTypeRdmp             = VirtualDiskType("rdmp")
	VirtualDiskTypeRaw              = VirtualDiskType("raw")
	VirtualDiskTypeDelta            = VirtualDiskType("delta")
	VirtualDiskTypeSparse2Gb        = VirtualDiskType("sparse2Gb")
	VirtualDiskTypeThick2Gb         = VirtualDiskType("thick2Gb")
	VirtualDiskTypeEagerZeroedThick = VirtualDiskType("eagerZeroedThick")
	VirtualDiskTypeSparseMonolithic = VirtualDiskType("sparseMonolithic")
	VirtualDiskTypeFlatMonolithic   = VirtualDiskType("flatMonolithic")
	VirtualDiskTypeThick            = VirtualDiskType("thick")
)

func init() {
	t["VirtualDiskType"] = reflect.TypeOf((*VirtualDiskType)(nil)).Elem()
}

type VirtualDiskVFlashCacheConfigInfoCacheConsistencyType string

const (
	VirtualDiskVFlashCacheConfigInfoCacheConsistencyTypeStrong = VirtualDiskVFlashCacheConfigInfoCacheConsistencyType("strong")
	VirtualDiskVFlashCacheConfigInfoCacheConsistencyTypeWeak   = VirtualDiskVFlashCacheConfigInfoCacheConsistencyType("weak")
)

func init() {
	t["VirtualDiskVFlashCacheConfigInfoCacheConsistencyType"] = reflect.TypeOf((*VirtualDiskVFlashCacheConfigInfoCacheConsistencyType)(nil)).Elem()
}

type VirtualDiskVFlashCacheConfigInfoCacheMode string

const (
	VirtualDiskVFlashCacheConfigInfoCacheModeWrite_thru = VirtualDiskVFlashCacheConfigInfoCacheMode("write_thru")
	VirtualDiskVFlashCacheConfigInfoCacheModeWrite_back = VirtualDiskVFlashCacheConfigInfoCacheMode("write_back")
)

func init() {
	t["VirtualDiskVFlashCacheConfigInfoCacheMode"] = reflect.TypeOf((*VirtualDiskVFlashCacheConfigInfoCacheMode)(nil)).Elem()
}

type VirtualEthernetCardLegacyNetworkDeviceName string

const (
	VirtualEthernetCardLegacyNetworkDeviceNameBridged  = VirtualEthernetCardLegacyNetworkDeviceName("bridged")
	VirtualEthernetCardLegacyNetworkDeviceNameNat      = VirtualEthernetCardLegacyNetworkDeviceName("nat")
	VirtualEthernetCardLegacyNetworkDeviceNameHostonly = VirtualEthernetCardLegacyNetworkDeviceName("hostonly")
)

func init() {
	t["VirtualEthernetCardLegacyNetworkDeviceName"] = reflect.TypeOf((*VirtualEthernetCardLegacyNetworkDeviceName)(nil)).Elem()
}

type VirtualEthernetCardMacType string

const (
	VirtualEthernetCardMacTypeManual    = VirtualEthernetCardMacType("manual")
	VirtualEthernetCardMacTypeGenerated = VirtualEthernetCardMacType("generated")
	VirtualEthernetCardMacTypeAssigned  = VirtualEthernetCardMacType("assigned")
)

func init() {
	t["VirtualEthernetCardMacType"] = reflect.TypeOf((*VirtualEthernetCardMacType)(nil)).Elem()
}

type VirtualMachineAppHeartbeatStatusType string

const (
	VirtualMachineAppHeartbeatStatusTypeAppStatusGray  = VirtualMachineAppHeartbeatStatusType("appStatusGray")
	VirtualMachineAppHeartbeatStatusTypeAppStatusGreen = VirtualMachineAppHeartbeatStatusType("appStatusGreen")
	VirtualMachineAppHeartbeatStatusTypeAppStatusRed   = VirtualMachineAppHeartbeatStatusType("appStatusRed")
)

func init() {
	t["VirtualMachineAppHeartbeatStatusType"] = reflect.TypeOf((*VirtualMachineAppHeartbeatStatusType)(nil)).Elem()
}

type VirtualMachineBootOptionsNetworkBootProtocolType string

const (
	VirtualMachineBootOptionsNetworkBootProtocolTypeIpv4 = VirtualMachineBootOptionsNetworkBootProtocolType("ipv4")
	VirtualMachineBootOptionsNetworkBootProtocolTypeIpv6 = VirtualMachineBootOptionsNetworkBootProtocolType("ipv6")
)

func init() {
	t["VirtualMachineBootOptionsNetworkBootProtocolType"] = reflect.TypeOf((*VirtualMachineBootOptionsNetworkBootProtocolType)(nil)).Elem()
}

type VirtualMachineConfigInfoNpivWwnType string

const (
	VirtualMachineConfigInfoNpivWwnTypeVc       = VirtualMachineConfigInfoNpivWwnType("vc")
	VirtualMachineConfigInfoNpivWwnTypeHost     = VirtualMachineConfigInfoNpivWwnType("host")
	VirtualMachineConfigInfoNpivWwnTypeExternal = VirtualMachineConfigInfoNpivWwnType("external")
)

func init() {
	t["VirtualMachineConfigInfoNpivWwnType"] = reflect.TypeOf((*VirtualMachineConfigInfoNpivWwnType)(nil)).Elem()
}

type VirtualMachineConfigInfoSwapPlacementType string

const (
	VirtualMachineConfigInfoSwapPlacementTypeInherit     = VirtualMachineConfigInfoSwapPlacementType("inherit")
	VirtualMachineConfigInfoSwapPlacementTypeVmDirectory = VirtualMachineConfigInfoSwapPlacementType("vmDirectory")
	VirtualMachineConfigInfoSwapPlacementTypeHostLocal   = VirtualMachineConfigInfoSwapPlacementType("hostLocal")
)

func init() {
	t["VirtualMachineConfigInfoSwapPlacementType"] = reflect.TypeOf((*VirtualMachineConfigInfoSwapPlacementType)(nil)).Elem()
}

type VirtualMachineConfigSpecEncryptedVMotionModes string

const (
	VirtualMachineConfigSpecEncryptedVMotionModesDisabled      = VirtualMachineConfigSpecEncryptedVMotionModes("disabled")
	VirtualMachineConfigSpecEncryptedVMotionModesOpportunistic = VirtualMachineConfigSpecEncryptedVMotionModes("opportunistic")
	VirtualMachineConfigSpecEncryptedVMotionModesRequired      = VirtualMachineConfigSpecEncryptedVMotionModes("required")
)

func init() {
	t["VirtualMachineConfigSpecEncryptedVMotionModes"] = reflect.TypeOf((*VirtualMachineConfigSpecEncryptedVMotionModes)(nil)).Elem()
}

type VirtualMachineConfigSpecNpivWwnOp string

const (
	VirtualMachineConfigSpecNpivWwnOpGenerate = VirtualMachineConfigSpecNpivWwnOp("generate")
	VirtualMachineConfigSpecNpivWwnOpSet      = VirtualMachineConfigSpecNpivWwnOp("set")
	VirtualMachineConfigSpecNpivWwnOpRemove   = VirtualMachineConfigSpecNpivWwnOp("remove")
	VirtualMachineConfigSpecNpivWwnOpExtend   = VirtualMachineConfigSpecNpivWwnOp("extend")
)

func init() {
	t["VirtualMachineConfigSpecNpivWwnOp"] = reflect.TypeOf((*VirtualMachineConfigSpecNpivWwnOp)(nil)).Elem()
}

type VirtualMachineConnectionState string

const (
	VirtualMachineConnectionStateConnected    = VirtualMachineConnectionState("connected")
	VirtualMachineConnectionStateDisconnected = VirtualMachineConnectionState("disconnected")
	VirtualMachineConnectionStateOrphaned     = VirtualMachineConnectionState("orphaned")
	VirtualMachineConnectionStateInaccessible = VirtualMachineConnectionState("inaccessible")
	VirtualMachineConnectionStateInvalid      = VirtualMachineConnectionState("invalid")
)

func init() {
	t["VirtualMachineConnectionState"] = reflect.TypeOf((*VirtualMachineConnectionState)(nil)).Elem()
}

type VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonOther string

const (
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonOtherVmNptIncompatibleHost    = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonOther("vmNptIncompatibleHost")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonOtherVmNptIncompatibleNetwork = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonOther("vmNptIncompatibleNetwork")
)

func init() {
	t["VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonOther"] = reflect.TypeOf((*VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonOther)(nil)).Elem()
}

type VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm string

const (
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptIncompatibleGuest                      = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptIncompatibleGuest")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptIncompatibleGuestDriver                = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptIncompatibleGuestDriver")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptIncompatibleAdapterType                = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptIncompatibleAdapterType")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptDisabledOrDisconnectedAdapter          = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptDisabledOrDisconnectedAdapter")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptIncompatibleAdapterFeatures            = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptIncompatibleAdapterFeatures")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptIncompatibleBackingType                = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptIncompatibleBackingType")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptInsufficientMemoryReservation          = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptInsufficientMemoryReservation")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptFaultToleranceOrRecordReplayConfigured = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptFaultToleranceOrRecordReplayConfigured")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptConflictingIOChainConfigured           = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptConflictingIOChainConfigured")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptMonitorBlocks                          = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptMonitorBlocks")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptConflictingOperationInProgress         = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptConflictingOperationInProgress")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptRuntimeError                           = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptRuntimeError")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptOutOfIntrVector                        = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptOutOfIntrVector")
	VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVmVmNptVMCIActive                             = VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm("vmNptVMCIActive")
)

func init() {
	t["VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm"] = reflect.TypeOf((*VirtualMachineDeviceRuntimeInfoVirtualEthernetCardRuntimeStateVmDirectPathGen2InactiveReasonVm)(nil)).Elem()
}

type VirtualMachineFaultToleranceState string

const (
	VirtualMachineFaultToleranceStateNotConfigured = VirtualMachineFaultToleranceState("notConfigured")
	VirtualMachineFaultToleranceStateDisabled      = VirtualMachineFaultToleranceState("disabled")
	VirtualMachineFaultToleranceStateEnabled       = VirtualMachineFaultToleranceState("enabled")
	VirtualMachineFaultToleranceStateNeedSecondary = VirtualMachineFaultToleranceState("needSecondary")
	VirtualMachineFaultToleranceStateStarting      = VirtualMachineFaultToleranceState("starting")
	VirtualMachineFaultToleranceStateRunning       = VirtualMachineFaultToleranceState("running")
)

func init() {
	t["VirtualMachineFaultToleranceState"] = reflect.TypeOf((*VirtualMachineFaultToleranceState)(nil)).Elem()
}

type VirtualMachineFaultToleranceType string

const (
	VirtualMachineFaultToleranceTypeUnset         = VirtualMachineFaultToleranceType("unset")
	VirtualMachineFaultToleranceTypeRecordReplay  = VirtualMachineFaultToleranceType("recordReplay")
	VirtualMachineFaultToleranceTypeCheckpointing = VirtualMachineFaultToleranceType("checkpointing")
)

func init() {
	t["VirtualMachineFaultToleranceType"] = reflect.TypeOf((*VirtualMachineFaultToleranceType)(nil)).Elem()
}

type VirtualMachineFileLayoutExFileType string

const (
	VirtualMachineFileLayoutExFileTypeConfig               = VirtualMachineFileLayoutExFileType("config")
	VirtualMachineFileLayoutExFileTypeExtendedConfig       = VirtualMachineFileLayoutExFileType("extendedConfig")
	VirtualMachineFileLayoutExFileTypeDiskDescriptor       = VirtualMachineFileLayoutExFileType("diskDescriptor")
	VirtualMachineFileLayoutExFileTypeDiskExtent           = VirtualMachineFileLayoutExFileType("diskExtent")
	VirtualMachineFileLayoutExFileTypeDigestDescriptor     = VirtualMachineFileLayoutExFileType("digestDescriptor")
	VirtualMachineFileLayoutExFileTypeDigestExtent         = VirtualMachineFileLayoutExFileType("digestExtent")
	VirtualMachineFileLayoutExFileTypeDiskReplicationState = VirtualMachineFileLayoutExFileType("diskReplicationState")
	VirtualMachineFileLayoutExFileTypeLog                  = VirtualMachineFileLayoutExFileType("log")
	VirtualMachineFileLayoutExFileTypeStat                 = VirtualMachineFileLayoutExFileType("stat")
	VirtualMachineFileLayoutExFileTypeNamespaceData        = VirtualMachineFileLayoutExFileType("namespaceData")
	VirtualMachineFileLayoutExFileTypeNvram                = VirtualMachineFileLayoutExFileType("nvram")
	VirtualMachineFileLayoutExFileTypeSnapshotData         = VirtualMachineFileLayoutExFileType("snapshotData")
	VirtualMachineFileLayoutExFileTypeSnapshotMemory       = VirtualMachineFileLayoutExFileType("snapshotMemory")
	VirtualMachineFileLayoutExFileTypeSnapshotList         = VirtualMachineFileLayoutExFileType("snapshotList")
	VirtualMachineFileLayoutExFileTypeSnapshotManifestList = VirtualMachineFileLayoutExFileType("snapshotManifestList")
	VirtualMachineFileLayoutExFileTypeSuspend              = VirtualMachineFileLayoutExFileType("suspend")
	VirtualMachineFileLayoutExFileTypeSuspendMemory        = VirtualMachineFileLayoutExFileType("suspendMemory")
	VirtualMachineFileLayoutExFileTypeSwap                 = VirtualMachineFileLayoutExFileType("swap")
	VirtualMachineFileLayoutExFileTypeUwswap               = VirtualMachineFileLayoutExFileType("uwswap")
	VirtualMachineFileLayoutExFileTypeCore                 = VirtualMachineFileLayoutExFileType("core")
	VirtualMachineFileLayoutExFileTypeScreenshot           = VirtualMachineFileLayoutExFileType("screenshot")
	VirtualMachineFileLayoutExFileTypeFtMetadata           = VirtualMachineFileLayoutExFileType("ftMetadata")
	VirtualMachineFileLayoutExFileTypeGuestCustomization   = VirtualMachineFileLayoutExFileType("guestCustomization")
)

func init() {
	t["VirtualMachineFileLayoutExFileType"] = reflect.TypeOf((*VirtualMachineFileLayoutExFileType)(nil)).Elem()
}

type VirtualMachineFlagInfoMonitorType string

const (
	VirtualMachineFlagInfoMonitorTypeRelease = VirtualMachineFlagInfoMonitorType("release")
	VirtualMachineFlagInfoMonitorTypeDebug   = VirtualMachineFlagInfoMonitorType("debug")
	VirtualMachineFlagInfoMonitorTypeStats   = VirtualMachineFlagInfoMonitorType("stats")
)

func init() {
	t["VirtualMachineFlagInfoMonitorType"] = reflect.TypeOf((*VirtualMachineFlagInfoMonitorType)(nil)).Elem()
}

type VirtualMachineFlagInfoVirtualExecUsage string

const (
	VirtualMachineFlagInfoVirtualExecUsageHvAuto = VirtualMachineFlagInfoVirtualExecUsage("hvAuto")
	VirtualMachineFlagInfoVirtualExecUsageHvOn   = VirtualMachineFlagInfoVirtualExecUsage("hvOn")
	VirtualMachineFlagInfoVirtualExecUsageHvOff  = VirtualMachineFlagInfoVirtualExecUsage("hvOff")
)

func init() {
	t["VirtualMachineFlagInfoVirtualExecUsage"] = reflect.TypeOf((*VirtualMachineFlagInfoVirtualExecUsage)(nil)).Elem()
}

type VirtualMachineFlagInfoVirtualMmuUsage string

const (
	VirtualMachineFlagInfoVirtualMmuUsageAutomatic = VirtualMachineFlagInfoVirtualMmuUsage("automatic")
	VirtualMachineFlagInfoVirtualMmuUsageOn        = VirtualMachineFlagInfoVirtualMmuUsage("on")
	VirtualMachineFlagInfoVirtualMmuUsageOff       = VirtualMachineFlagInfoVirtualMmuUsage("off")
)

func init() {
	t["VirtualMachineFlagInfoVirtualMmuUsage"] = reflect.TypeOf((*VirtualMachineFlagInfoVirtualMmuUsage)(nil)).Elem()
}

type VirtualMachineForkConfigInfoChildType string

const (
	VirtualMachineForkConfigInfoChildTypeNone          = VirtualMachineForkConfigInfoChildType("none")
	VirtualMachineForkConfigInfoChildTypePersistent    = VirtualMachineForkConfigInfoChildType("persistent")
	VirtualMachineForkConfigInfoChildTypeNonpersistent = VirtualMachineForkConfigInfoChildType("nonpersistent")
)

func init() {
	t["VirtualMachineForkConfigInfoChildType"] = reflect.TypeOf((*VirtualMachineForkConfigInfoChildType)(nil)).Elem()
}

type VirtualMachineGuestOsFamily string

const (
	VirtualMachineGuestOsFamilyWindowsGuest      = VirtualMachineGuestOsFamily("windowsGuest")
	VirtualMachineGuestOsFamilyLinuxGuest        = VirtualMachineGuestOsFamily("linuxGuest")
	VirtualMachineGuestOsFamilyNetwareGuest      = VirtualMachineGuestOsFamily("netwareGuest")
	VirtualMachineGuestOsFamilySolarisGuest      = VirtualMachineGuestOsFamily("solarisGuest")
	VirtualMachineGuestOsFamilyDarwinGuestFamily = VirtualMachineGuestOsFamily("darwinGuestFamily")
	VirtualMachineGuestOsFamilyOtherGuestFamily  = VirtualMachineGuestOsFamily("otherGuestFamily")
)

func init() {
	t["VirtualMachineGuestOsFamily"] = reflect.TypeOf((*VirtualMachineGuestOsFamily)(nil)).Elem()
}

type VirtualMachineGuestOsIdentifier string

const (
	VirtualMachineGuestOsIdentifierDosGuest                = VirtualMachineGuestOsIdentifier("dosGuest")
	VirtualMachineGuestOsIdentifierWin31Guest              = VirtualMachineGuestOsIdentifier("win31Guest")
	VirtualMachineGuestOsIdentifierWin95Guest              = VirtualMachineGuestOsIdentifier("win95Guest")
	VirtualMachineGuestOsIdentifierWin98Guest              = VirtualMachineGuestOsIdentifier("win98Guest")
	VirtualMachineGuestOsIdentifierWinMeGuest              = VirtualMachineGuestOsIdentifier("winMeGuest")
	VirtualMachineGuestOsIdentifierWinNTGuest              = VirtualMachineGuestOsIdentifier("winNTGuest")
	VirtualMachineGuestOsIdentifierWin2000ProGuest         = VirtualMachineGuestOsIdentifier("win2000ProGuest")
	VirtualMachineGuestOsIdentifierWin2000ServGuest        = VirtualMachineGuestOsIdentifier("win2000ServGuest")
	VirtualMachineGuestOsIdentifierWin2000AdvServGuest     = VirtualMachineGuestOsIdentifier("win2000AdvServGuest")
	VirtualMachineGuestOsIdentifierWinXPHomeGuest          = VirtualMachineGuestOsIdentifier("winXPHomeGuest")
	VirtualMachineGuestOsIdentifierWinXPProGuest           = VirtualMachineGuestOsIdentifier("winXPProGuest")
	VirtualMachineGuestOsIdentifierWinXPPro64Guest         = VirtualMachineGuestOsIdentifier("winXPPro64Guest")
	VirtualMachineGuestOsIdentifierWinNetWebGuest          = VirtualMachineGuestOsIdentifier("winNetWebGuest")
	VirtualMachineGuestOsIdentifierWinNetStandardGuest     = VirtualMachineGuestOsIdentifier("winNetStandardGuest")
	VirtualMachineGuestOsIdentifierWinNetEnterpriseGuest   = VirtualMachineGuestOsIdentifier("winNetEnterpriseGuest")
	VirtualMachineGuestOsIdentifierWinNetDatacenterGuest   = VirtualMachineGuestOsIdentifier("winNetDatacenterGuest")
	VirtualMachineGuestOsIdentifierWinNetBusinessGuest     = VirtualMachineGuestOsIdentifier("winNetBusinessGuest")
	VirtualMachineGuestOsIdentifierWinNetStandard64Guest   = VirtualMachineGuestOsIdentifier("winNetStandard64Guest")
	VirtualMachineGuestOsIdentifierWinNetEnterprise64Guest = VirtualMachineGuestOsIdentifier("winNetEnterprise64Guest")
	VirtualMachineGuestOsIdentifierWinLonghornGuest        = VirtualMachineGuestOsIdentifier("winLonghornGuest")
	VirtualMachineGuestOsIdentifierWinLonghorn64Guest      = VirtualMachineGuestOsIdentifier("winLonghorn64Guest")
	VirtualMachineGuestOsIdentifierWinNetDatacenter64Guest = VirtualMachineGuestOsIdentifier("winNetDatacenter64Guest")
	VirtualMachineGuestOsIdentifierWinVistaGuest           = VirtualMachineGuestOsIdentifier("winVistaGuest")
	VirtualMachineGuestOsIdentifierWinVista64Guest         = VirtualMachineGuestOsIdentifier("winVista64Guest")
	VirtualMachineGuestOsIdentifierWindows7Guest           = VirtualMachineGuestOsIdentifier("windows7Guest")
	VirtualMachineGuestOsIdentifierWindows7_64Guest        = VirtualMachineGuestOsIdentifier("windows7_64Guest")
	VirtualMachineGuestOsIdentifierWindows7Server64Guest   = VirtualMachineGuestOsIdentifier("windows7Server64Guest")
	VirtualMachineGuestOsIdentifierWindows8Guest           = VirtualMachineGuestOsIdentifier("windows8Guest")
	VirtualMachineGuestOsIdentifierWindows8_64Guest        = VirtualMachineGuestOsIdentifier("windows8_64Guest")
	VirtualMachineGuestOsIdentifierWindows8Server64Guest   = VirtualMachineGuestOsIdentifier("windows8Server64Guest")
	VirtualMachineGuestOsIdentifierWindows9Guest           = VirtualMachineGuestOsIdentifier("windows9Guest")
	VirtualMachineGuestOsIdentifierWindows9_64Guest        = VirtualMachineGuestOsIdentifier("windows9_64Guest")
	VirtualMachineGuestOsIdentifierWindows9Server64Guest   = VirtualMachineGuestOsIdentifier("windows9Server64Guest")
	VirtualMachineGuestOsIdentifierWindowsHyperVGuest      = VirtualMachineGuestOsIdentifier("windowsHyperVGuest")
	VirtualMachineGuestOsIdentifierFreebsdGuest            = VirtualMachineGuestOsIdentifier("freebsdGuest")
	VirtualMachineGuestOsIdentifierFreebsd64Guest          = VirtualMachineGuestOsIdentifier("freebsd64Guest")
	VirtualMachineGuestOsIdentifierRedhatGuest             = VirtualMachineGuestOsIdentifier("redhatGuest")
	VirtualMachineGuestOsIdentifierRhel2Guest              = VirtualMachineGuestOsIdentifier("rhel2Guest")
	VirtualMachineGuestOsIdentifierRhel3Guest              = VirtualMachineGuestOsIdentifier("rhel3Guest")
	VirtualMachineGuestOsIdentifierRhel3_64Guest           = VirtualMachineGuestOsIdentifier("rhel3_64Guest")
	VirtualMachineGuestOsIdentifierRhel4Guest              = VirtualMachineGuestOsIdentifier("rhel4Guest")
	VirtualMachineGuestOsIdentifierRhel4_64Guest           = VirtualMachineGuestOsIdentifier("rhel4_64Guest")
	VirtualMachineGuestOsIdentifierRhel5Guest              = VirtualMachineGuestOsIdentifier("rhel5Guest")
	VirtualMachineGuestOsIdentifierRhel5_64Guest           = VirtualMachineGuestOsIdentifier("rhel5_64Guest")
	VirtualMachineGuestOsIdentifierRhel6Guest              = VirtualMachineGuestOsIdentifier("rhel6Guest")
	VirtualMachineGuestOsIdentifierRhel6_64Guest           = VirtualMachineGuestOsIdentifier("rhel6_64Guest")
	VirtualMachineGuestOsIdentifierRhel7Guest              = VirtualMachineGuestOsIdentifier("rhel7Guest")
	VirtualMachineGuestOsIdentifierRhel7_64Guest           = VirtualMachineGuestOsIdentifier("rhel7_64Guest")
	VirtualMachineGuestOsIdentifierCentosGuest             = VirtualMachineGuestOsIdentifier("centosGuest")
	VirtualMachineGuestOsIdentifierCentos64Guest           = VirtualMachineGuestOsIdentifier("centos64Guest")
	VirtualMachineGuestOsIdentifierCentos6Guest            = VirtualMachineGuestOsIdentifier("centos6Guest")
	VirtualMachineGuestOsIdentifierCentos6_64Guest         = VirtualMachineGuestOsIdentifier("centos6_64Guest")
	VirtualMachineGuestOsIdentifierCentos7Guest            = VirtualMachineGuestOsIdentifier("centos7Guest")
	VirtualMachineGuestOsIdentifierCentos7_64Guest         = VirtualMachineGuestOsIdentifier("centos7_64Guest")
	VirtualMachineGuestOsIdentifierOracleLinuxGuest        = VirtualMachineGuestOsIdentifier("oracleLinuxGuest")
	VirtualMachineGuestOsIdentifierOracleLinux64Guest      = VirtualMachineGuestOsIdentifier("oracleLinux64Guest")
	VirtualMachineGuestOsIdentifierOracleLinux6Guest       = VirtualMachineGuestOsIdentifier("oracleLinux6Guest")
	VirtualMachineGuestOsIdentifierOracleLinux6_64Guest    = VirtualMachineGuestOsIdentifier("oracleLinux6_64Guest")
	VirtualMachineGuestOsIdentifierOracleLinux7Guest       = VirtualMachineGuestOsIdentifier("oracleLinux7Guest")
	VirtualMachineGuestOsIdentifierOracleLinux7_64Guest    = VirtualMachineGuestOsIdentifier("oracleLinux7_64Guest")
	VirtualMachineGuestOsIdentifierSuseGuest               = VirtualMachineGuestOsIdentifier("suseGuest")
	VirtualMachineGuestOsIdentifierSuse64Guest             = VirtualMachineGuestOsIdentifier("suse64Guest")
	VirtualMachineGuestOsIdentifierSlesGuest               = VirtualMachineGuestOsIdentifier("slesGuest")
	VirtualMachineGuestOsIdentifierSles64Guest             = VirtualMachineGuestOsIdentifier("sles64Guest")
	VirtualMachineGuestOsIdentifierSles10Guest             = VirtualMachineGuestOsIdentifier("sles10Guest")
	VirtualMachineGuestOsIdentifierSles10_64Guest          = VirtualMachineGuestOsIdentifier("sles10_64Guest")
	VirtualMachineGuestOsIdentifierSles11Guest             = VirtualMachineGuestOsIdentifier("sles11Guest")
	VirtualMachineGuestOsIdentifierSles11_64Guest          = VirtualMachineGuestOsIdentifier("sles11_64Guest")
	VirtualMachineGuestOsIdentifierSles12Guest             = VirtualMachineGuestOsIdentifier("sles12Guest")
	VirtualMachineGuestOsIdentifierSles12_64Guest          = VirtualMachineGuestOsIdentifier("sles12_64Guest")
	VirtualMachineGuestOsIdentifierNld9Guest               = VirtualMachineGuestOsIdentifier("nld9Guest")
	VirtualMachineGuestOsIdentifierOesGuest                = VirtualMachineGuestOsIdentifier("oesGuest")
	VirtualMachineGuestOsIdentifierSjdsGuest               = VirtualMachineGuestOsIdentifier("sjdsGuest")
	VirtualMachineGuestOsIdentifierMandrakeGuest           = VirtualMachineGuestOsIdentifier("mandrakeGuest")
	VirtualMachineGuestOsIdentifierMandrivaGuest           = VirtualMachineGuestOsIdentifier("mandrivaGuest")
	VirtualMachineGuestOsIdentifierMandriva64Guest         = VirtualMachineGuestOsIdentifier("mandriva64Guest")
	VirtualMachineGuestOsIdentifierTurboLinuxGuest         = VirtualMachineGuestOsIdentifier("turboLinuxGuest")
	VirtualMachineGuestOsIdentifierTurboLinux64Guest       = VirtualMachineGuestOsIdentifier("turboLinux64Guest")
	VirtualMachineGuestOsIdentifierUbuntuGuest             = VirtualMachineGuestOsIdentifier("ubuntuGuest")
	VirtualMachineGuestOsIdentifierUbuntu64Guest           = VirtualMachineGuestOsIdentifier("ubuntu64Guest")
	VirtualMachineGuestOsIdentifierDebian4Guest            = VirtualMachineGuestOsIdentifier("debian4Guest")
	VirtualMachineGuestOsIdentifierDebian4_64Guest         = VirtualMachineGuestOsIdentifier("debian4_64Guest")
	VirtualMachineGuestOsIdentifierDebian5Guest            = VirtualMachineGuestOsIdentifier("debian5Guest")
	VirtualMachineGuestOsIdentifierDebian5_64Guest         = VirtualMachineGuestOsIdentifier("debian5_64Guest")
	VirtualMachineGuestOsIdentifierDebian6Guest            = VirtualMachineGuestOsIdentifier("debian6Guest")
	VirtualMachineGuestOsIdentifierDebian6_64Guest         = VirtualMachineGuestOsIdentifier("debian6_64Guest")
	VirtualMachineGuestOsIdentifierDebian7Guest            = VirtualMachineGuestOsIdentifier("debian7Guest")
	VirtualMachineGuestOsIdentifierDebian7_64Guest         = VirtualMachineGuestOsIdentifier("debian7_64Guest")
	VirtualMachineGuestOsIdentifierDebian8Guest            = VirtualMachineGuestOsIdentifier("debian8Guest")
	VirtualMachineGuestOsIdentifierDebian8_64Guest         = VirtualMachineGuestOsIdentifier("debian8_64Guest")
	VirtualMachineGuestOsIdentifierDebian9Guest            = VirtualMachineGuestOsIdentifier("debian9Guest")
	VirtualMachineGuestOsIdentifierDebian9_64Guest         = VirtualMachineGuestOsIdentifier("debian9_64Guest")
	VirtualMachineGuestOsIdentifierDebian10Guest           = VirtualMachineGuestOsIdentifier("debian10Guest")
	VirtualMachineGuestOsIdentifierDebian10_64Guest        = VirtualMachineGuestOsIdentifier("debian10_64Guest")
	VirtualMachineGuestOsIdentifierAsianux3Guest           = VirtualMachineGuestOsIdentifier("asianux3Guest")
	VirtualMachineGuestOsIdentifierAsianux3_64Guest        = VirtualMachineGuestOsIdentifier("asianux3_64Guest")
	VirtualMachineGuestOsIdentifierAsianux4Guest           = VirtualMachineGuestOsIdentifier("asianux4Guest")
	VirtualMachineGuestOsIdentifierAsianux4_64Guest        = VirtualMachineGuestOsIdentifier("asianux4_64Guest")
	VirtualMachineGuestOsIdentifierAsianux5_64Guest        = VirtualMachineGuestOsIdentifier("asianux5_64Guest")
	VirtualMachineGuestOsIdentifierAsianux7_64Guest        = VirtualMachineGuestOsIdentifier("asianux7_64Guest")
	VirtualMachineGuestOsIdentifierOpensuseGuest           = VirtualMachineGuestOsIdentifier("opensuseGuest")
	VirtualMachineGuestOsIdentifierOpensuse64Guest         = VirtualMachineGuestOsIdentifier("opensuse64Guest")
	VirtualMachineGuestOsIdentifierFedoraGuest             = VirtualMachineGuestOsIdentifier("fedoraGuest")
	VirtualMachineGuestOsIdentifierFedora64Guest           = VirtualMachineGuestOsIdentifier("fedora64Guest")
	VirtualMachineGuestOsIdentifierCoreos64Guest           = VirtualMachineGuestOsIdentifier("coreos64Guest")
	VirtualMachineGuestOsIdentifierVmwarePhoton64Guest     = VirtualMachineGuestOsIdentifier("vmwarePhoton64Guest")
	VirtualMachineGuestOsIdentifierOther24xLinuxGuest      = VirtualMachineGuestOsIdentifier("other24xLinuxGuest")
	VirtualMachineGuestOsIdentifierOther26xLinuxGuest      = VirtualMachineGuestOsIdentifier("other26xLinuxGuest")
	VirtualMachineGuestOsIdentifierOtherLinuxGuest         = VirtualMachineGuestOsIdentifier("otherLinuxGuest")
	VirtualMachineGuestOsIdentifierOther3xLinuxGuest       = VirtualMachineGuestOsIdentifier("other3xLinuxGuest")
	VirtualMachineGuestOsIdentifierGenericLinuxGuest       = VirtualMachineGuestOsIdentifier("genericLinuxGuest")
	VirtualMachineGuestOsIdentifierOther24xLinux64Guest    = VirtualMachineGuestOsIdentifier("other24xLinux64Guest")
	VirtualMachineGuestOsIdentifierOther26xLinux64Guest    = VirtualMachineGuestOsIdentifier("other26xLinux64Guest")
	VirtualMachineGuestOsIdentifierOther3xLinux64Guest     = VirtualMachineGuestOsIdentifier("other3xLinux64Guest")
	VirtualMachineGuestOsIdentifierOtherLinux64Guest       = VirtualMachineGuestOsIdentifier("otherLinux64Guest")
	VirtualMachineGuestOsIdentifierSolaris6Guest           = VirtualMachineGuestOsIdentifier("solaris6Guest")
	VirtualMachineGuestOsIdentifierSolaris7Guest           = VirtualMachineGuestOsIdentifier("solaris7Guest")
	VirtualMachineGuestOsIdentifierSolaris8Guest           = VirtualMachineGuestOsIdentifier("solaris8Guest")
	VirtualMachineGuestOsIdentifierSolaris9Guest           = VirtualMachineGuestOsIdentifier("solaris9Guest")
	VirtualMachineGuestOsIdentifierSolaris10Guest          = VirtualMachineGuestOsIdentifier("solaris10Guest")
	VirtualMachineGuestOsIdentifierSolaris10_64Guest       = VirtualMachineGuestOsIdentifier("solaris10_64Guest")
	VirtualMachineGuestOsIdentifierSolaris11_64Guest       = VirtualMachineGuestOsIdentifier("solaris11_64Guest")
	VirtualMachineGuestOsIdentifierOs2Guest                = VirtualMachineGuestOsIdentifier("os2Guest")
	VirtualMachineGuestOsIdentifierEComStationGuest        = VirtualMachineGuestOsIdentifier("eComStationGuest")
	VirtualMachineGuestOsIdentifierEComStation2Guest       = VirtualMachineGuestOsIdentifier("eComStation2Guest")
	VirtualMachineGuestOsIdentifierNetware4Guest           = VirtualMachineGuestOsIdentifier("netware4Guest")
	VirtualMachineGuestOsIdentifierNetware5Guest           = VirtualMachineGuestOsIdentifier("netware5Guest")
	VirtualMachineGuestOsIdentifierNetware6Guest           = VirtualMachineGuestOsIdentifier("netware6Guest")
	VirtualMachineGuestOsIdentifierOpenServer5Guest        = VirtualMachineGuestOsIdentifier("openServer5Guest")
	VirtualMachineGuestOsIdentifierOpenServer6Guest        = VirtualMachineGuestOsIdentifier("openServer6Guest")
	VirtualMachineGuestOsIdentifierUnixWare7Guest          = VirtualMachineGuestOsIdentifier("unixWare7Guest")
	VirtualMachineGuestOsIdentifierDarwinGuest             = VirtualMachineGuestOsIdentifier("darwinGuest")
	VirtualMachineGuestOsIdentifierDarwin64Guest           = VirtualMachineGuestOsIdentifier("darwin64Guest")
	VirtualMachineGuestOsIdentifierDarwin10Guest           = VirtualMachineGuestOsIdentifier("darwin10Guest")
	VirtualMachineGuestOsIdentifierDarwin10_64Guest        = VirtualMachineGuestOsIdentifier("darwin10_64Guest")
	VirtualMachineGuestOsIdentifierDarwin11Guest           = VirtualMachineGuestOsIdentifier("darwin11Guest")
	VirtualMachineGuestOsIdentifierDarwin11_64Guest        = VirtualMachineGuestOsIdentifier("darwin11_64Guest")
	VirtualMachineGuestOsIdentifierDarwin12_64Guest        = VirtualMachineGuestOsIdentifier("darwin12_64Guest")
	VirtualMachineGuestOsIdentifierDarwin13_64Guest        = VirtualMachineGuestOsIdentifier("darwin13_64Guest")
	VirtualMachineGuestOsIdentifierDarwin14_64Guest        = VirtualMachineGuestOsIdentifier("darwin14_64Guest")
	VirtualMachineGuestOsIdentifierDarwin15_64Guest        = VirtualMachineGuestOsIdentifier("darwin15_64Guest")
	VirtualMachineGuestOsIdentifierDarwin16_64Guest        = VirtualMachineGuestOsIdentifier("darwin16_64Guest")
	VirtualMachineGuestOsIdentifierVmkernelGuest           = VirtualMachineGuestOsIdentifier("vmkernelGuest")
	VirtualMachineGuestOsIdentifierVmkernel5Guest          = VirtualMachineGuestOsIdentifier("vmkernel5Guest")
	VirtualMachineGuestOsIdentifierVmkernel6Guest          = VirtualMachineGuestOsIdentifier("vmkernel6Guest")
	VirtualMachineGuestOsIdentifierVmkernel65Guest         = VirtualMachineGuestOsIdentifier("vmkernel65Guest")
	VirtualMachineGuestOsIdentifierOtherGuest              = VirtualMachineGuestOsIdentifier("otherGuest")
	VirtualMachineGuestOsIdentifierOtherGuest64            = VirtualMachineGuestOsIdentifier("otherGuest64")
)

func init() {
	t["VirtualMachineGuestOsIdentifier"] = reflect.TypeOf((*VirtualMachineGuestOsIdentifier)(nil)).Elem()
}

type VirtualMachineGuestState string

const (
	VirtualMachineGuestStateRunning      = VirtualMachineGuestState("running")
	VirtualMachineGuestStateShuttingDown = VirtualMachineGuestState("shuttingDown")
	VirtualMachineGuestStateResetting    = VirtualMachineGuestState("resetting")
	VirtualMachineGuestStateStandby      = VirtualMachineGuestState("standby")
	VirtualMachineGuestStateNotRunning   = VirtualMachineGuestState("notRunning")
	VirtualMachineGuestStateUnknown      = VirtualMachineGuestState("unknown")
)

func init() {
	t["VirtualMachineGuestState"] = reflect.TypeOf((*VirtualMachineGuestState)(nil)).Elem()
}

type VirtualMachineHtSharing string

const (
	VirtualMachineHtSharingAny      = VirtualMachineHtSharing("any")
	VirtualMachineHtSharingNone     = VirtualMachineHtSharing("none")
	VirtualMachineHtSharingInternal = VirtualMachineHtSharing("internal")
)

func init() {
	t["VirtualMachineHtSharing"] = reflect.TypeOf((*VirtualMachineHtSharing)(nil)).Elem()
}

type VirtualMachineMemoryAllocationPolicy string

const (
	VirtualMachineMemoryAllocationPolicySwapNone = VirtualMachineMemoryAllocationPolicy("swapNone")
	VirtualMachineMemoryAllocationPolicySwapSome = VirtualMachineMemoryAllocationPolicy("swapSome")
	VirtualMachineMemoryAllocationPolicySwapMost = VirtualMachineMemoryAllocationPolicy("swapMost")
)

func init() {
	t["VirtualMachineMemoryAllocationPolicy"] = reflect.TypeOf((*VirtualMachineMemoryAllocationPolicy)(nil)).Elem()
}

type VirtualMachineMetadataManagerVmMetadataOp string

const (
	VirtualMachineMetadataManagerVmMetadataOpUpdate = VirtualMachineMetadataManagerVmMetadataOp("Update")
	VirtualMachineMetadataManagerVmMetadataOpRemove = VirtualMachineMetadataManagerVmMetadataOp("Remove")
)

func init() {
	t["VirtualMachineMetadataManagerVmMetadataOp"] = reflect.TypeOf((*VirtualMachineMetadataManagerVmMetadataOp)(nil)).Elem()
}

type VirtualMachineMetadataManagerVmMetadataOwnerOwner string

const (
	VirtualMachineMetadataManagerVmMetadataOwnerOwnerComVmwareVsphereHA = VirtualMachineMetadataManagerVmMetadataOwnerOwner("ComVmwareVsphereHA")
)

func init() {
	t["VirtualMachineMetadataManagerVmMetadataOwnerOwner"] = reflect.TypeOf((*VirtualMachineMetadataManagerVmMetadataOwnerOwner)(nil)).Elem()
}

type VirtualMachineMovePriority string

const (
	VirtualMachineMovePriorityLowPriority     = VirtualMachineMovePriority("lowPriority")
	VirtualMachineMovePriorityHighPriority    = VirtualMachineMovePriority("highPriority")
	VirtualMachineMovePriorityDefaultPriority = VirtualMachineMovePriority("defaultPriority")
)

func init() {
	t["VirtualMachineMovePriority"] = reflect.TypeOf((*VirtualMachineMovePriority)(nil)).Elem()
}

type VirtualMachineNeedSecondaryReason string

const (
	VirtualMachineNeedSecondaryReasonInitializing           = VirtualMachineNeedSecondaryReason("initializing")
	VirtualMachineNeedSecondaryReasonDivergence             = VirtualMachineNeedSecondaryReason("divergence")
	VirtualMachineNeedSecondaryReasonLostConnection         = VirtualMachineNeedSecondaryReason("lostConnection")
	VirtualMachineNeedSecondaryReasonPartialHardwareFailure = VirtualMachineNeedSecondaryReason("partialHardwareFailure")
	VirtualMachineNeedSecondaryReasonUserAction             = VirtualMachineNeedSecondaryReason("userAction")
	VirtualMachineNeedSecondaryReasonCheckpointError        = VirtualMachineNeedSecondaryReason("checkpointError")
	VirtualMachineNeedSecondaryReasonOther                  = VirtualMachineNeedSecondaryReason("other")
)

func init() {
	t["VirtualMachineNeedSecondaryReason"] = reflect.TypeOf((*VirtualMachineNeedSecondaryReason)(nil)).Elem()
}

type VirtualMachinePowerOffBehavior string

const (
	VirtualMachinePowerOffBehaviorPowerOff = VirtualMachinePowerOffBehavior("powerOff")
	VirtualMachinePowerOffBehaviorRevert   = VirtualMachinePowerOffBehavior("revert")
	VirtualMachinePowerOffBehaviorPrompt   = VirtualMachinePowerOffBehavior("prompt")
	VirtualMachinePowerOffBehaviorTake     = VirtualMachinePowerOffBehavior("take")
)

func init() {
	t["VirtualMachinePowerOffBehavior"] = reflect.TypeOf((*VirtualMachinePowerOffBehavior)(nil)).Elem()
}

type VirtualMachinePowerOpType string

const (
	VirtualMachinePowerOpTypeSoft   = VirtualMachinePowerOpType("soft")
	VirtualMachinePowerOpTypeHard   = VirtualMachinePowerOpType("hard")
	VirtualMachinePowerOpTypePreset = VirtualMachinePowerOpType("preset")
)

func init() {
	t["VirtualMachinePowerOpType"] = reflect.TypeOf((*VirtualMachinePowerOpType)(nil)).Elem()
}

type VirtualMachinePowerState string

const (
	VirtualMachinePowerStatePoweredOff = VirtualMachinePowerState("poweredOff")
	VirtualMachinePowerStatePoweredOn  = VirtualMachinePowerState("poweredOn")
	VirtualMachinePowerStateSuspended  = VirtualMachinePowerState("suspended")
)

func init() {
	t["VirtualMachinePowerState"] = reflect.TypeOf((*VirtualMachinePowerState)(nil)).Elem()
}

type VirtualMachineRecordReplayState string

const (
	VirtualMachineRecordReplayStateRecording = VirtualMachineRecordReplayState("recording")
	VirtualMachineRecordReplayStateReplaying = VirtualMachineRecordReplayState("replaying")
	VirtualMachineRecordReplayStateInactive  = VirtualMachineRecordReplayState("inactive")
)

func init() {
	t["VirtualMachineRecordReplayState"] = reflect.TypeOf((*VirtualMachineRecordReplayState)(nil)).Elem()
}

type VirtualMachineRelocateDiskMoveOptions string

const (
	VirtualMachineRelocateDiskMoveOptionsMoveAllDiskBackingsAndAllowSharing    = VirtualMachineRelocateDiskMoveOptions("moveAllDiskBackingsAndAllowSharing")
	VirtualMachineRelocateDiskMoveOptionsMoveAllDiskBackingsAndDisallowSharing = VirtualMachineRelocateDiskMoveOptions("moveAllDiskBackingsAndDisallowSharing")
	VirtualMachineRelocateDiskMoveOptionsMoveChildMostDiskBacking              = VirtualMachineRelocateDiskMoveOptions("moveChildMostDiskBacking")
	VirtualMachineRelocateDiskMoveOptionsCreateNewChildDiskBacking             = VirtualMachineRelocateDiskMoveOptions("createNewChildDiskBacking")
	VirtualMachineRelocateDiskMoveOptionsMoveAllDiskBackingsAndConsolidate     = VirtualMachineRelocateDiskMoveOptions("moveAllDiskBackingsAndConsolidate")
)

func init() {
	t["VirtualMachineRelocateDiskMoveOptions"] = reflect.TypeOf((*VirtualMachineRelocateDiskMoveOptions)(nil)).Elem()
}

type VirtualMachineRelocateTransformation string

const (
	VirtualMachineRelocateTransformationFlat   = VirtualMachineRelocateTransformation("flat")
	VirtualMachineRelocateTransformationSparse = VirtualMachineRelocateTransformation("sparse")
)

func init() {
	t["VirtualMachineRelocateTransformation"] = reflect.TypeOf((*VirtualMachineRelocateTransformation)(nil)).Elem()
}

type VirtualMachineScsiPassthroughType string

const (
	VirtualMachineScsiPassthroughTypeDisk      = VirtualMachineScsiPassthroughType("disk")
	VirtualMachineScsiPassthroughTypeTape      = VirtualMachineScsiPassthroughType("tape")
	VirtualMachineScsiPassthroughTypePrinter   = VirtualMachineScsiPassthroughType("printer")
	VirtualMachineScsiPassthroughTypeProcessor = VirtualMachineScsiPassthroughType("processor")
	VirtualMachineScsiPassthroughTypeWorm      = VirtualMachineScsiPassthroughType("worm")
	VirtualMachineScsiPassthroughTypeCdrom     = VirtualMachineScsiPassthroughType("cdrom")
	VirtualMachineScsiPassthroughTypeScanner   = VirtualMachineScsiPassthroughType("scanner")
	VirtualMachineScsiPassthroughTypeOptical   = VirtualMachineScsiPassthroughType("optical")
	VirtualMachineScsiPassthroughTypeMedia     = VirtualMachineScsiPassthroughType("media")
	VirtualMachineScsiPassthroughTypeCom       = VirtualMachineScsiPassthroughType("com")
	VirtualMachineScsiPassthroughTypeRaid      = VirtualMachineScsiPassthroughType("raid")
	VirtualMachineScsiPassthroughTypeUnknown   = VirtualMachineScsiPassthroughType("unknown")
)

func init() {
	t["VirtualMachineScsiPassthroughType"] = reflect.TypeOf((*VirtualMachineScsiPassthroughType)(nil)).Elem()
}

type VirtualMachineStandbyActionType string

const (
	VirtualMachineStandbyActionTypeCheckpoint     = VirtualMachineStandbyActionType("checkpoint")
	VirtualMachineStandbyActionTypePowerOnSuspend = VirtualMachineStandbyActionType("powerOnSuspend")
)

func init() {
	t["VirtualMachineStandbyActionType"] = reflect.TypeOf((*VirtualMachineStandbyActionType)(nil)).Elem()
}

type VirtualMachineTargetInfoConfigurationTag string

const (
	VirtualMachineTargetInfoConfigurationTagCompliant   = VirtualMachineTargetInfoConfigurationTag("compliant")
	VirtualMachineTargetInfoConfigurationTagClusterWide = VirtualMachineTargetInfoConfigurationTag("clusterWide")
)

func init() {
	t["VirtualMachineTargetInfoConfigurationTag"] = reflect.TypeOf((*VirtualMachineTargetInfoConfigurationTag)(nil)).Elem()
}

type VirtualMachineTicketType string

const (
	VirtualMachineTicketTypeMks          = VirtualMachineTicketType("mks")
	VirtualMachineTicketTypeDevice       = VirtualMachineTicketType("device")
	VirtualMachineTicketTypeGuestControl = VirtualMachineTicketType("guestControl")
	VirtualMachineTicketTypeWebmks       = VirtualMachineTicketType("webmks")
)

func init() {
	t["VirtualMachineTicketType"] = reflect.TypeOf((*VirtualMachineTicketType)(nil)).Elem()
}

type VirtualMachineToolsInstallType string

const (
	VirtualMachineToolsInstallTypeGuestToolsTypeUnknown     = VirtualMachineToolsInstallType("guestToolsTypeUnknown")
	VirtualMachineToolsInstallTypeGuestToolsTypeMSI         = VirtualMachineToolsInstallType("guestToolsTypeMSI")
	VirtualMachineToolsInstallTypeGuestToolsTypeTar         = VirtualMachineToolsInstallType("guestToolsTypeTar")
	VirtualMachineToolsInstallTypeGuestToolsTypeOSP         = VirtualMachineToolsInstallType("guestToolsTypeOSP")
	VirtualMachineToolsInstallTypeGuestToolsTypeOpenVMTools = VirtualMachineToolsInstallType("guestToolsTypeOpenVMTools")
)

func init() {
	t["VirtualMachineToolsInstallType"] = reflect.TypeOf((*VirtualMachineToolsInstallType)(nil)).Elem()
}

type VirtualMachineToolsRunningStatus string

const (
	VirtualMachineToolsRunningStatusGuestToolsNotRunning       = VirtualMachineToolsRunningStatus("guestToolsNotRunning")
	VirtualMachineToolsRunningStatusGuestToolsRunning          = VirtualMachineToolsRunningStatus("guestToolsRunning")
	VirtualMachineToolsRunningStatusGuestToolsExecutingScripts = VirtualMachineToolsRunningStatus("guestToolsExecutingScripts")
)

func init() {
	t["VirtualMachineToolsRunningStatus"] = reflect.TypeOf((*VirtualMachineToolsRunningStatus)(nil)).Elem()
}

type VirtualMachineToolsStatus string

const (
	VirtualMachineToolsStatusToolsNotInstalled = VirtualMachineToolsStatus("toolsNotInstalled")
	VirtualMachineToolsStatusToolsNotRunning   = VirtualMachineToolsStatus("toolsNotRunning")
	VirtualMachineToolsStatusToolsOld          = VirtualMachineToolsStatus("toolsOld")
	VirtualMachineToolsStatusToolsOk           = VirtualMachineToolsStatus("toolsOk")
)

func init() {
	t["VirtualMachineToolsStatus"] = reflect.TypeOf((*VirtualMachineToolsStatus)(nil)).Elem()
}

type VirtualMachineToolsVersionStatus string

const (
	VirtualMachineToolsVersionStatusGuestToolsNotInstalled = VirtualMachineToolsVersionStatus("guestToolsNotInstalled")
	VirtualMachineToolsVersionStatusGuestToolsNeedUpgrade  = VirtualMachineToolsVersionStatus("guestToolsNeedUpgrade")
	VirtualMachineToolsVersionStatusGuestToolsCurrent      = VirtualMachineToolsVersionStatus("guestToolsCurrent")
	VirtualMachineToolsVersionStatusGuestToolsUnmanaged    = VirtualMachineToolsVersionStatus("guestToolsUnmanaged")
	VirtualMachineToolsVersionStatusGuestToolsTooOld       = VirtualMachineToolsVersionStatus("guestToolsTooOld")
	VirtualMachineToolsVersionStatusGuestToolsSupportedOld = VirtualMachineToolsVersionStatus("guestToolsSupportedOld")
	VirtualMachineToolsVersionStatusGuestToolsSupportedNew = VirtualMachineToolsVersionStatus("guestToolsSupportedNew")
	VirtualMachineToolsVersionStatusGuestToolsTooNew       = VirtualMachineToolsVersionStatus("guestToolsTooNew")
	VirtualMachineToolsVersionStatusGuestToolsBlacklisted  = VirtualMachineToolsVersionStatus("guestToolsBlacklisted")
)

func init() {
	t["VirtualMachineToolsVersionStatus"] = reflect.TypeOf((*VirtualMachineToolsVersionStatus)(nil)).Elem()
}

type VirtualMachineUsbInfoFamily string

const (
	VirtualMachineUsbInfoFamilyAudio           = VirtualMachineUsbInfoFamily("audio")
	VirtualMachineUsbInfoFamilyHid             = VirtualMachineUsbInfoFamily("hid")
	VirtualMachineUsbInfoFamilyHid_bootable    = VirtualMachineUsbInfoFamily("hid_bootable")
	VirtualMachineUsbInfoFamilyPhysical        = VirtualMachineUsbInfoFamily("physical")
	VirtualMachineUsbInfoFamilyCommunication   = VirtualMachineUsbInfoFamily("communication")
	VirtualMachineUsbInfoFamilyImaging         = VirtualMachineUsbInfoFamily("imaging")
	VirtualMachineUsbInfoFamilyPrinter         = VirtualMachineUsbInfoFamily("printer")
	VirtualMachineUsbInfoFamilyStorage         = VirtualMachineUsbInfoFamily("storage")
	VirtualMachineUsbInfoFamilyHub             = VirtualMachineUsbInfoFamily("hub")
	VirtualMachineUsbInfoFamilySmart_card      = VirtualMachineUsbInfoFamily("smart_card")
	VirtualMachineUsbInfoFamilySecurity        = VirtualMachineUsbInfoFamily("security")
	VirtualMachineUsbInfoFamilyVideo           = VirtualMachineUsbInfoFamily("video")
	VirtualMachineUsbInfoFamilyWireless        = VirtualMachineUsbInfoFamily("wireless")
	VirtualMachineUsbInfoFamilyBluetooth       = VirtualMachineUsbInfoFamily("bluetooth")
	VirtualMachineUsbInfoFamilyWusb            = VirtualMachineUsbInfoFamily("wusb")
	VirtualMachineUsbInfoFamilyPda             = VirtualMachineUsbInfoFamily("pda")
	VirtualMachineUsbInfoFamilyVendor_specific = VirtualMachineUsbInfoFamily("vendor_specific")
	VirtualMachineUsbInfoFamilyOther           = VirtualMachineUsbInfoFamily("other")
	VirtualMachineUsbInfoFamilyUnknownFamily   = VirtualMachineUsbInfoFamily("unknownFamily")
)

func init() {
	t["VirtualMachineUsbInfoFamily"] = reflect.TypeOf((*VirtualMachineUsbInfoFamily)(nil)).Elem()
}

type VirtualMachineUsbInfoSpeed string

const (
	VirtualMachineUsbInfoSpeedLow          = VirtualMachineUsbInfoSpeed("low")
	VirtualMachineUsbInfoSpeedFull         = VirtualMachineUsbInfoSpeed("full")
	VirtualMachineUsbInfoSpeedHigh         = VirtualMachineUsbInfoSpeed("high")
	VirtualMachineUsbInfoSpeedSuperSpeed   = VirtualMachineUsbInfoSpeed("superSpeed")
	VirtualMachineUsbInfoSpeedUnknownSpeed = VirtualMachineUsbInfoSpeed("unknownSpeed")
)

func init() {
	t["VirtualMachineUsbInfoSpeed"] = reflect.TypeOf((*VirtualMachineUsbInfoSpeed)(nil)).Elem()
}

type VirtualMachineVMCIDeviceAction string

const (
	VirtualMachineVMCIDeviceActionAllow = VirtualMachineVMCIDeviceAction("allow")
	VirtualMachineVMCIDeviceActionDeny  = VirtualMachineVMCIDeviceAction("deny")
)

func init() {
	t["VirtualMachineVMCIDeviceAction"] = reflect.TypeOf((*VirtualMachineVMCIDeviceAction)(nil)).Elem()
}

type VirtualMachineVMCIDeviceDirection string

const (
	VirtualMachineVMCIDeviceDirectionGuest        = VirtualMachineVMCIDeviceDirection("guest")
	VirtualMachineVMCIDeviceDirectionHost         = VirtualMachineVMCIDeviceDirection("host")
	VirtualMachineVMCIDeviceDirectionAnyDirection = VirtualMachineVMCIDeviceDirection("anyDirection")
)

func init() {
	t["VirtualMachineVMCIDeviceDirection"] = reflect.TypeOf((*VirtualMachineVMCIDeviceDirection)(nil)).Elem()
}

type VirtualMachineVMCIDeviceProtocol string

const (
	VirtualMachineVMCIDeviceProtocolHypervisor  = VirtualMachineVMCIDeviceProtocol("hypervisor")
	VirtualMachineVMCIDeviceProtocolDoorbell    = VirtualMachineVMCIDeviceProtocol("doorbell")
	VirtualMachineVMCIDeviceProtocolQueuepair   = VirtualMachineVMCIDeviceProtocol("queuepair")
	VirtualMachineVMCIDeviceProtocolDatagram    = VirtualMachineVMCIDeviceProtocol("datagram")
	VirtualMachineVMCIDeviceProtocolStream      = VirtualMachineVMCIDeviceProtocol("stream")
	VirtualMachineVMCIDeviceProtocolAnyProtocol = VirtualMachineVMCIDeviceProtocol("anyProtocol")
)

func init() {
	t["VirtualMachineVMCIDeviceProtocol"] = reflect.TypeOf((*VirtualMachineVMCIDeviceProtocol)(nil)).Elem()
}

type VirtualMachineVideoCardUse3dRenderer string

const (
	VirtualMachineVideoCardUse3dRendererAutomatic = VirtualMachineVideoCardUse3dRenderer("automatic")
	VirtualMachineVideoCardUse3dRendererSoftware  = VirtualMachineVideoCardUse3dRenderer("software")
	VirtualMachineVideoCardUse3dRendererHardware  = VirtualMachineVideoCardUse3dRenderer("hardware")
)

func init() {
	t["VirtualMachineVideoCardUse3dRenderer"] = reflect.TypeOf((*VirtualMachineVideoCardUse3dRenderer)(nil)).Elem()
}

type VirtualMachineWindowsQuiesceSpecVssBackupContext string

const (
	VirtualMachineWindowsQuiesceSpecVssBackupContextCtx_auto              = VirtualMachineWindowsQuiesceSpecVssBackupContext("ctx_auto")
	VirtualMachineWindowsQuiesceSpecVssBackupContextCtx_backup            = VirtualMachineWindowsQuiesceSpecVssBackupContext("ctx_backup")
	VirtualMachineWindowsQuiesceSpecVssBackupContextCtx_file_share_backup = VirtualMachineWindowsQuiesceSpecVssBackupContext("ctx_file_share_backup")
)

func init() {
	t["VirtualMachineWindowsQuiesceSpecVssBackupContext"] = reflect.TypeOf((*VirtualMachineWindowsQuiesceSpecVssBackupContext)(nil)).Elem()
}

type VirtualPointingDeviceHostChoice string

const (
	VirtualPointingDeviceHostChoiceAutodetect           = VirtualPointingDeviceHostChoice("autodetect")
	VirtualPointingDeviceHostChoiceIntellimouseExplorer = VirtualPointingDeviceHostChoice("intellimouseExplorer")
	VirtualPointingDeviceHostChoiceIntellimousePs2      = VirtualPointingDeviceHostChoice("intellimousePs2")
	VirtualPointingDeviceHostChoiceLogitechMouseman     = VirtualPointingDeviceHostChoice("logitechMouseman")
	VirtualPointingDeviceHostChoiceMicrosoft_serial     = VirtualPointingDeviceHostChoice("microsoft_serial")
	VirtualPointingDeviceHostChoiceMouseSystems         = VirtualPointingDeviceHostChoice("mouseSystems")
	VirtualPointingDeviceHostChoiceMousemanSerial       = VirtualPointingDeviceHostChoice("mousemanSerial")
	VirtualPointingDeviceHostChoicePs2                  = VirtualPointingDeviceHostChoice("ps2")
)

func init() {
	t["VirtualPointingDeviceHostChoice"] = reflect.TypeOf((*VirtualPointingDeviceHostChoice)(nil)).Elem()
}

type VirtualSCSISharing string

const (
	VirtualSCSISharingNoSharing       = VirtualSCSISharing("noSharing")
	VirtualSCSISharingVirtualSharing  = VirtualSCSISharing("virtualSharing")
	VirtualSCSISharingPhysicalSharing = VirtualSCSISharing("physicalSharing")
)

func init() {
	t["VirtualSCSISharing"] = reflect.TypeOf((*VirtualSCSISharing)(nil)).Elem()
}

type VirtualSerialPortEndPoint string

const (
	VirtualSerialPortEndPointClient = VirtualSerialPortEndPoint("client")
	VirtualSerialPortEndPointServer = VirtualSerialPortEndPoint("server")
)

func init() {
	t["VirtualSerialPortEndPoint"] = reflect.TypeOf((*VirtualSerialPortEndPoint)(nil)).Elem()
}

type VmDasBeingResetEventReasonCode string

const (
	VmDasBeingResetEventReasonCodeVmtoolsHeartbeatFailure  = VmDasBeingResetEventReasonCode("vmtoolsHeartbeatFailure")
	VmDasBeingResetEventReasonCodeAppHeartbeatFailure      = VmDasBeingResetEventReasonCode("appHeartbeatFailure")
	VmDasBeingResetEventReasonCodeAppImmediateResetRequest = VmDasBeingResetEventReasonCode("appImmediateResetRequest")
	VmDasBeingResetEventReasonCodeVmcpResetApdCleared      = VmDasBeingResetEventReasonCode("vmcpResetApdCleared")
)

func init() {
	t["VmDasBeingResetEventReasonCode"] = reflect.TypeOf((*VmDasBeingResetEventReasonCode)(nil)).Elem()
}

type VmFailedStartingSecondaryEventFailureReason string

const (
	VmFailedStartingSecondaryEventFailureReasonIncompatibleHost = VmFailedStartingSecondaryEventFailureReason("incompatibleHost")
	VmFailedStartingSecondaryEventFailureReasonLoginFailed      = VmFailedStartingSecondaryEventFailureReason("loginFailed")
	VmFailedStartingSecondaryEventFailureReasonRegisterVmFailed = VmFailedStartingSecondaryEventFailureReason("registerVmFailed")
	VmFailedStartingSecondaryEventFailureReasonMigrateFailed    = VmFailedStartingSecondaryEventFailureReason("migrateFailed")
)

func init() {
	t["VmFailedStartingSecondaryEventFailureReason"] = reflect.TypeOf((*VmFailedStartingSecondaryEventFailureReason)(nil)).Elem()
}

type VmFaultToleranceConfigIssueReasonForIssue string

const (
	VmFaultToleranceConfigIssueReasonForIssueHaNotEnabled                   = VmFaultToleranceConfigIssueReasonForIssue("haNotEnabled")
	VmFaultToleranceConfigIssueReasonForIssueMoreThanOneSecondary           = VmFaultToleranceConfigIssueReasonForIssue("moreThanOneSecondary")
	VmFaultToleranceConfigIssueReasonForIssueRecordReplayNotSupported       = VmFaultToleranceConfigIssueReasonForIssue("recordReplayNotSupported")
	VmFaultToleranceConfigIssueReasonForIssueReplayNotSupported             = VmFaultToleranceConfigIssueReasonForIssue("replayNotSupported")
	VmFaultToleranceConfigIssueReasonForIssueTemplateVm                     = VmFaultToleranceConfigIssueReasonForIssue("templateVm")
	VmFaultToleranceConfigIssueReasonForIssueMultipleVCPU                   = VmFaultToleranceConfigIssueReasonForIssue("multipleVCPU")
	VmFaultToleranceConfigIssueReasonForIssueHostInactive                   = VmFaultToleranceConfigIssueReasonForIssue("hostInactive")
	VmFaultToleranceConfigIssueReasonForIssueFtUnsupportedHardware          = VmFaultToleranceConfigIssueReasonForIssue("ftUnsupportedHardware")
	VmFaultToleranceConfigIssueReasonForIssueFtUnsupportedProduct           = VmFaultToleranceConfigIssueReasonForIssue("ftUnsupportedProduct")
	VmFaultToleranceConfigIssueReasonForIssueMissingVMotionNic              = VmFaultToleranceConfigIssueReasonForIssue("missingVMotionNic")
	VmFaultToleranceConfigIssueReasonForIssueMissingFTLoggingNic            = VmFaultToleranceConfigIssueReasonForIssue("missingFTLoggingNic")
	VmFaultToleranceConfigIssueReasonForIssueThinDisk                       = VmFaultToleranceConfigIssueReasonForIssue("thinDisk")
	VmFaultToleranceConfigIssueReasonForIssueVerifySSLCertificateFlagNotSet = VmFaultToleranceConfigIssueReasonForIssue("verifySSLCertificateFlagNotSet")
	VmFaultToleranceConfigIssueReasonForIssueHasSnapshots                   = VmFaultToleranceConfigIssueReasonForIssue("hasSnapshots")
	VmFaultToleranceConfigIssueReasonForIssueNoConfig                       = VmFaultToleranceConfigIssueReasonForIssue("noConfig")
	VmFaultToleranceConfigIssueReasonForIssueFtSecondaryVm                  = VmFaultToleranceConfigIssueReasonForIssue("ftSecondaryVm")
	VmFaultToleranceConfigIssueReasonForIssueHasLocalDisk                   = VmFaultToleranceConfigIssueReasonForIssue("hasLocalDisk")
	VmFaultToleranceConfigIssueReasonForIssueEsxAgentVm                     = VmFaultToleranceConfigIssueReasonForIssue("esxAgentVm")
	VmFaultToleranceConfigIssueReasonForIssueVideo3dEnabled                 = VmFaultToleranceConfigIssueReasonForIssue("video3dEnabled")
	VmFaultToleranceConfigIssueReasonForIssueHasUnsupportedDisk             = VmFaultToleranceConfigIssueReasonForIssue("hasUnsupportedDisk")
	VmFaultToleranceConfigIssueReasonForIssueInsufficientBandwidth          = VmFaultToleranceConfigIssueReasonForIssue("insufficientBandwidth")
	VmFaultToleranceConfigIssueReasonForIssueHasNestedHVConfiguration       = VmFaultToleranceConfigIssueReasonForIssue("hasNestedHVConfiguration")
	VmFaultToleranceConfigIssueReasonForIssueHasVFlashConfiguration         = VmFaultToleranceConfigIssueReasonForIssue("hasVFlashConfiguration")
	VmFaultToleranceConfigIssueReasonForIssueUnsupportedProduct             = VmFaultToleranceConfigIssueReasonForIssue("unsupportedProduct")
	VmFaultToleranceConfigIssueReasonForIssueCpuHvUnsupported               = VmFaultToleranceConfigIssueReasonForIssue("cpuHvUnsupported")
	VmFaultToleranceConfigIssueReasonForIssueCpuHwmmuUnsupported            = VmFaultToleranceConfigIssueReasonForIssue("cpuHwmmuUnsupported")
	VmFaultToleranceConfigIssueReasonForIssueCpuHvDisabled                  = VmFaultToleranceConfigIssueReasonForIssue("cpuHvDisabled")
	VmFaultToleranceConfigIssueReasonForIssueHasEFIFirmware                 = VmFaultToleranceConfigIssueReasonForIssue("hasEFIFirmware")
)

func init() {
	t["VmFaultToleranceConfigIssueReasonForIssue"] = reflect.TypeOf((*VmFaultToleranceConfigIssueReasonForIssue)(nil)).Elem()
}

type VmFaultToleranceInvalidFileBackingDeviceType string

const (
	VmFaultToleranceInvalidFileBackingDeviceTypeVirtualFloppy       = VmFaultToleranceInvalidFileBackingDeviceType("virtualFloppy")
	VmFaultToleranceInvalidFileBackingDeviceTypeVirtualCdrom        = VmFaultToleranceInvalidFileBackingDeviceType("virtualCdrom")
	VmFaultToleranceInvalidFileBackingDeviceTypeVirtualSerialPort   = VmFaultToleranceInvalidFileBackingDeviceType("virtualSerialPort")
	VmFaultToleranceInvalidFileBackingDeviceTypeVirtualParallelPort = VmFaultToleranceInvalidFileBackingDeviceType("virtualParallelPort")
	VmFaultToleranceInvalidFileBackingDeviceTypeVirtualDisk         = VmFaultToleranceInvalidFileBackingDeviceType("virtualDisk")
)

func init() {
	t["VmFaultToleranceInvalidFileBackingDeviceType"] = reflect.TypeOf((*VmFaultToleranceInvalidFileBackingDeviceType)(nil)).Elem()
}

type VmShutdownOnIsolationEventOperation string

const (
	VmShutdownOnIsolationEventOperationShutdown   = VmShutdownOnIsolationEventOperation("shutdown")
	VmShutdownOnIsolationEventOperationPoweredOff = VmShutdownOnIsolationEventOperation("poweredOff")
)

func init() {
	t["VmShutdownOnIsolationEventOperation"] = reflect.TypeOf((*VmShutdownOnIsolationEventOperation)(nil)).Elem()
}

type VmwareDistributedVirtualSwitchPvlanPortType string

const (
	VmwareDistributedVirtualSwitchPvlanPortTypePromiscuous = VmwareDistributedVirtualSwitchPvlanPortType("promiscuous")
	VmwareDistributedVirtualSwitchPvlanPortTypeIsolated    = VmwareDistributedVirtualSwitchPvlanPortType("isolated")
	VmwareDistributedVirtualSwitchPvlanPortTypeCommunity   = VmwareDistributedVirtualSwitchPvlanPortType("community")
)

func init() {
	t["VmwareDistributedVirtualSwitchPvlanPortType"] = reflect.TypeOf((*VmwareDistributedVirtualSwitchPvlanPortType)(nil)).Elem()
}

type VsanDiskIssueType string

const (
	VsanDiskIssueTypeNonExist      = VsanDiskIssueType("nonExist")
	VsanDiskIssueTypeStampMismatch = VsanDiskIssueType("stampMismatch")
	VsanDiskIssueTypeUnknown       = VsanDiskIssueType("unknown")
)

func init() {
	t["VsanDiskIssueType"] = reflect.TypeOf((*VsanDiskIssueType)(nil)).Elem()
}

type VsanHostDecommissionModeObjectAction string

const (
	VsanHostDecommissionModeObjectActionNoAction                  = VsanHostDecommissionModeObjectAction("noAction")
	VsanHostDecommissionModeObjectActionEnsureObjectAccessibility = VsanHostDecommissionModeObjectAction("ensureObjectAccessibility")
	VsanHostDecommissionModeObjectActionEvacuateAllData           = VsanHostDecommissionModeObjectAction("evacuateAllData")
)

func init() {
	t["VsanHostDecommissionModeObjectAction"] = reflect.TypeOf((*VsanHostDecommissionModeObjectAction)(nil)).Elem()
}

type VsanHostDiskResultState string

const (
	VsanHostDiskResultStateInUse      = VsanHostDiskResultState("inUse")
	VsanHostDiskResultStateEligible   = VsanHostDiskResultState("eligible")
	VsanHostDiskResultStateIneligible = VsanHostDiskResultState("ineligible")
)

func init() {
	t["VsanHostDiskResultState"] = reflect.TypeOf((*VsanHostDiskResultState)(nil)).Elem()
}

type VsanHostHealthState string

const (
	VsanHostHealthStateUnknown   = VsanHostHealthState("unknown")
	VsanHostHealthStateHealthy   = VsanHostHealthState("healthy")
	VsanHostHealthStateUnhealthy = VsanHostHealthState("unhealthy")
)

func init() {
	t["VsanHostHealthState"] = reflect.TypeOf((*VsanHostHealthState)(nil)).Elem()
}

type VsanHostNodeState string

const (
	VsanHostNodeStateError                   = VsanHostNodeState("error")
	VsanHostNodeStateDisabled                = VsanHostNodeState("disabled")
	VsanHostNodeStateAgent                   = VsanHostNodeState("agent")
	VsanHostNodeStateMaster                  = VsanHostNodeState("master")
	VsanHostNodeStateBackup                  = VsanHostNodeState("backup")
	VsanHostNodeStateStarting                = VsanHostNodeState("starting")
	VsanHostNodeStateStopping                = VsanHostNodeState("stopping")
	VsanHostNodeStateEnteringMaintenanceMode = VsanHostNodeState("enteringMaintenanceMode")
	VsanHostNodeStateExitingMaintenanceMode  = VsanHostNodeState("exitingMaintenanceMode")
	VsanHostNodeStateDecommissioning         = VsanHostNodeState("decommissioning")
)

func init() {
	t["VsanHostNodeState"] = reflect.TypeOf((*VsanHostNodeState)(nil)).Elem()
}

type VsanUpgradeSystemUpgradeHistoryDiskGroupOpType string

const (
	VsanUpgradeSystemUpgradeHistoryDiskGroupOpTypeAdd    = VsanUpgradeSystemUpgradeHistoryDiskGroupOpType("add")
	VsanUpgradeSystemUpgradeHistoryDiskGroupOpTypeRemove = VsanUpgradeSystemUpgradeHistoryDiskGroupOpType("remove")
)

func init() {
	t["VsanUpgradeSystemUpgradeHistoryDiskGroupOpType"] = reflect.TypeOf((*VsanUpgradeSystemUpgradeHistoryDiskGroupOpType)(nil)).Elem()
}

type WeekOfMonth string

const (
	WeekOfMonthFirst  = WeekOfMonth("first")
	WeekOfMonthSecond = WeekOfMonth("second")
	WeekOfMonthThird  = WeekOfMonth("third")
	WeekOfMonthFourth = WeekOfMonth("fourth")
	WeekOfMonthLast   = WeekOfMonth("last")
)

func init() {
	t["WeekOfMonth"] = reflect.TypeOf((*WeekOfMonth)(nil)).Elem()
}

type WillLoseHAProtectionResolution string

const (
	WillLoseHAProtectionResolutionSvmotion = WillLoseHAProtectionResolution("svmotion")
	WillLoseHAProtectionResolutionRelocate = WillLoseHAProtectionResolution("relocate")
)

func init() {
	t["WillLoseHAProtectionResolution"] = reflect.TypeOf((*WillLoseHAProtectionResolution)(nil)).Elem()
}
