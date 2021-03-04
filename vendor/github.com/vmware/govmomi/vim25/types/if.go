/*
Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.

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

func (b *Action) GetAction() *Action { return b }

type BaseAction interface {
	GetAction() *Action
}

func init() {
	t["BaseAction"] = reflect.TypeOf((*Action)(nil)).Elem()
}

func (b *ActiveDirectoryFault) GetActiveDirectoryFault() *ActiveDirectoryFault { return b }

type BaseActiveDirectoryFault interface {
	GetActiveDirectoryFault() *ActiveDirectoryFault
}

func init() {
	t["BaseActiveDirectoryFault"] = reflect.TypeOf((*ActiveDirectoryFault)(nil)).Elem()
}

func (b *AlarmAction) GetAlarmAction() *AlarmAction { return b }

type BaseAlarmAction interface {
	GetAlarmAction() *AlarmAction
}

func init() {
	t["BaseAlarmAction"] = reflect.TypeOf((*AlarmAction)(nil)).Elem()
}

func (b *AlarmEvent) GetAlarmEvent() *AlarmEvent { return b }

type BaseAlarmEvent interface {
	GetAlarmEvent() *AlarmEvent
}

func init() {
	t["BaseAlarmEvent"] = reflect.TypeOf((*AlarmEvent)(nil)).Elem()
}

func (b *AlarmExpression) GetAlarmExpression() *AlarmExpression { return b }

type BaseAlarmExpression interface {
	GetAlarmExpression() *AlarmExpression
}

func init() {
	t["BaseAlarmExpression"] = reflect.TypeOf((*AlarmExpression)(nil)).Elem()
}

func (b *AlarmSpec) GetAlarmSpec() *AlarmSpec { return b }

type BaseAlarmSpec interface {
	GetAlarmSpec() *AlarmSpec
}

func init() {
	t["BaseAlarmSpec"] = reflect.TypeOf((*AlarmSpec)(nil)).Elem()
}

func (b *AnswerFileCreateSpec) GetAnswerFileCreateSpec() *AnswerFileCreateSpec { return b }

type BaseAnswerFileCreateSpec interface {
	GetAnswerFileCreateSpec() *AnswerFileCreateSpec
}

func init() {
	t["BaseAnswerFileCreateSpec"] = reflect.TypeOf((*AnswerFileCreateSpec)(nil)).Elem()
}

func (b *ApplyProfile) GetApplyProfile() *ApplyProfile { return b }

type BaseApplyProfile interface {
	GetApplyProfile() *ApplyProfile
}

func init() {
	t["BaseApplyProfile"] = reflect.TypeOf((*ApplyProfile)(nil)).Elem()
}

func (b *ArrayUpdateSpec) GetArrayUpdateSpec() *ArrayUpdateSpec { return b }

type BaseArrayUpdateSpec interface {
	GetArrayUpdateSpec() *ArrayUpdateSpec
}

func init() {
	t["BaseArrayUpdateSpec"] = reflect.TypeOf((*ArrayUpdateSpec)(nil)).Elem()
}

func (b *AuthorizationEvent) GetAuthorizationEvent() *AuthorizationEvent { return b }

type BaseAuthorizationEvent interface {
	GetAuthorizationEvent() *AuthorizationEvent
}

func init() {
	t["BaseAuthorizationEvent"] = reflect.TypeOf((*AuthorizationEvent)(nil)).Elem()
}

func (b *BaseConfigInfo) GetBaseConfigInfo() *BaseConfigInfo { return b }

type BaseBaseConfigInfo interface {
	GetBaseConfigInfo() *BaseConfigInfo
}

func init() {
	t["BaseBaseConfigInfo"] = reflect.TypeOf((*BaseConfigInfo)(nil)).Elem()
}

func (b *BaseConfigInfoBackingInfo) GetBaseConfigInfoBackingInfo() *BaseConfigInfoBackingInfo {
	return b
}

type BaseBaseConfigInfoBackingInfo interface {
	GetBaseConfigInfoBackingInfo() *BaseConfigInfoBackingInfo
}

func init() {
	t["BaseBaseConfigInfoBackingInfo"] = reflect.TypeOf((*BaseConfigInfoBackingInfo)(nil)).Elem()
}

func (b *BaseConfigInfoFileBackingInfo) GetBaseConfigInfoFileBackingInfo() *BaseConfigInfoFileBackingInfo {
	return b
}

type BaseBaseConfigInfoFileBackingInfo interface {
	GetBaseConfigInfoFileBackingInfo() *BaseConfigInfoFileBackingInfo
}

func init() {
	t["BaseBaseConfigInfoFileBackingInfo"] = reflect.TypeOf((*BaseConfigInfoFileBackingInfo)(nil)).Elem()
}

func (b *CannotAccessNetwork) GetCannotAccessNetwork() *CannotAccessNetwork { return b }

type BaseCannotAccessNetwork interface {
	GetCannotAccessNetwork() *CannotAccessNetwork
}

func init() {
	t["BaseCannotAccessNetwork"] = reflect.TypeOf((*CannotAccessNetwork)(nil)).Elem()
}

func (b *CannotAccessVmComponent) GetCannotAccessVmComponent() *CannotAccessVmComponent { return b }

type BaseCannotAccessVmComponent interface {
	GetCannotAccessVmComponent() *CannotAccessVmComponent
}

func init() {
	t["BaseCannotAccessVmComponent"] = reflect.TypeOf((*CannotAccessVmComponent)(nil)).Elem()
}

func (b *CannotAccessVmDevice) GetCannotAccessVmDevice() *CannotAccessVmDevice { return b }

type BaseCannotAccessVmDevice interface {
	GetCannotAccessVmDevice() *CannotAccessVmDevice
}

func init() {
	t["BaseCannotAccessVmDevice"] = reflect.TypeOf((*CannotAccessVmDevice)(nil)).Elem()
}

func (b *CannotAccessVmDisk) GetCannotAccessVmDisk() *CannotAccessVmDisk { return b }

type BaseCannotAccessVmDisk interface {
	GetCannotAccessVmDisk() *CannotAccessVmDisk
}

func init() {
	t["BaseCannotAccessVmDisk"] = reflect.TypeOf((*CannotAccessVmDisk)(nil)).Elem()
}

func (b *CannotMoveVsanEnabledHost) GetCannotMoveVsanEnabledHost() *CannotMoveVsanEnabledHost {
	return b
}

type BaseCannotMoveVsanEnabledHost interface {
	GetCannotMoveVsanEnabledHost() *CannotMoveVsanEnabledHost
}

func init() {
	t["BaseCannotMoveVsanEnabledHost"] = reflect.TypeOf((*CannotMoveVsanEnabledHost)(nil)).Elem()
}

func (b *ClusterAction) GetClusterAction() *ClusterAction { return b }

type BaseClusterAction interface {
	GetClusterAction() *ClusterAction
}

func init() {
	t["BaseClusterAction"] = reflect.TypeOf((*ClusterAction)(nil)).Elem()
}

func (b *ClusterDasAdmissionControlInfo) GetClusterDasAdmissionControlInfo() *ClusterDasAdmissionControlInfo {
	return b
}

type BaseClusterDasAdmissionControlInfo interface {
	GetClusterDasAdmissionControlInfo() *ClusterDasAdmissionControlInfo
}

func init() {
	t["BaseClusterDasAdmissionControlInfo"] = reflect.TypeOf((*ClusterDasAdmissionControlInfo)(nil)).Elem()
}

func (b *ClusterDasAdmissionControlPolicy) GetClusterDasAdmissionControlPolicy() *ClusterDasAdmissionControlPolicy {
	return b
}

type BaseClusterDasAdmissionControlPolicy interface {
	GetClusterDasAdmissionControlPolicy() *ClusterDasAdmissionControlPolicy
}

func init() {
	t["BaseClusterDasAdmissionControlPolicy"] = reflect.TypeOf((*ClusterDasAdmissionControlPolicy)(nil)).Elem()
}

func (b *ClusterDasAdvancedRuntimeInfo) GetClusterDasAdvancedRuntimeInfo() *ClusterDasAdvancedRuntimeInfo {
	return b
}

type BaseClusterDasAdvancedRuntimeInfo interface {
	GetClusterDasAdvancedRuntimeInfo() *ClusterDasAdvancedRuntimeInfo
}

func init() {
	t["BaseClusterDasAdvancedRuntimeInfo"] = reflect.TypeOf((*ClusterDasAdvancedRuntimeInfo)(nil)).Elem()
}

func (b *ClusterDasData) GetClusterDasData() *ClusterDasData { return b }

type BaseClusterDasData interface {
	GetClusterDasData() *ClusterDasData
}

func init() {
	t["BaseClusterDasData"] = reflect.TypeOf((*ClusterDasData)(nil)).Elem()
}

func (b *ClusterDasHostInfo) GetClusterDasHostInfo() *ClusterDasHostInfo { return b }

type BaseClusterDasHostInfo interface {
	GetClusterDasHostInfo() *ClusterDasHostInfo
}

func init() {
	t["BaseClusterDasHostInfo"] = reflect.TypeOf((*ClusterDasHostInfo)(nil)).Elem()
}

func (b *ClusterDrsFaultsFaultsByVm) GetClusterDrsFaultsFaultsByVm() *ClusterDrsFaultsFaultsByVm {
	return b
}

type BaseClusterDrsFaultsFaultsByVm interface {
	GetClusterDrsFaultsFaultsByVm() *ClusterDrsFaultsFaultsByVm
}

func init() {
	t["BaseClusterDrsFaultsFaultsByVm"] = reflect.TypeOf((*ClusterDrsFaultsFaultsByVm)(nil)).Elem()
}

func (b *ClusterEvent) GetClusterEvent() *ClusterEvent { return b }

type BaseClusterEvent interface {
	GetClusterEvent() *ClusterEvent
}

func init() {
	t["BaseClusterEvent"] = reflect.TypeOf((*ClusterEvent)(nil)).Elem()
}

func (b *ClusterGroupInfo) GetClusterGroupInfo() *ClusterGroupInfo { return b }

type BaseClusterGroupInfo interface {
	GetClusterGroupInfo() *ClusterGroupInfo
}

func init() {
	t["BaseClusterGroupInfo"] = reflect.TypeOf((*ClusterGroupInfo)(nil)).Elem()
}

func (b *ClusterOvercommittedEvent) GetClusterOvercommittedEvent() *ClusterOvercommittedEvent {
	return b
}

type BaseClusterOvercommittedEvent interface {
	GetClusterOvercommittedEvent() *ClusterOvercommittedEvent
}

func init() {
	t["BaseClusterOvercommittedEvent"] = reflect.TypeOf((*ClusterOvercommittedEvent)(nil)).Elem()
}

func (b *ClusterProfileConfigSpec) GetClusterProfileConfigSpec() *ClusterProfileConfigSpec { return b }

type BaseClusterProfileConfigSpec interface {
	GetClusterProfileConfigSpec() *ClusterProfileConfigSpec
}

func init() {
	t["BaseClusterProfileConfigSpec"] = reflect.TypeOf((*ClusterProfileConfigSpec)(nil)).Elem()
}

func (b *ClusterProfileCreateSpec) GetClusterProfileCreateSpec() *ClusterProfileCreateSpec { return b }

type BaseClusterProfileCreateSpec interface {
	GetClusterProfileCreateSpec() *ClusterProfileCreateSpec
}

func init() {
	t["BaseClusterProfileCreateSpec"] = reflect.TypeOf((*ClusterProfileCreateSpec)(nil)).Elem()
}

func (b *ClusterRuleInfo) GetClusterRuleInfo() *ClusterRuleInfo { return b }

type BaseClusterRuleInfo interface {
	GetClusterRuleInfo() *ClusterRuleInfo
}

func init() {
	t["BaseClusterRuleInfo"] = reflect.TypeOf((*ClusterRuleInfo)(nil)).Elem()
}

func (b *ClusterSlotPolicy) GetClusterSlotPolicy() *ClusterSlotPolicy { return b }

type BaseClusterSlotPolicy interface {
	GetClusterSlotPolicy() *ClusterSlotPolicy
}

func init() {
	t["BaseClusterSlotPolicy"] = reflect.TypeOf((*ClusterSlotPolicy)(nil)).Elem()
}

func (b *ClusterStatusChangedEvent) GetClusterStatusChangedEvent() *ClusterStatusChangedEvent {
	return b
}

type BaseClusterStatusChangedEvent interface {
	GetClusterStatusChangedEvent() *ClusterStatusChangedEvent
}

func init() {
	t["BaseClusterStatusChangedEvent"] = reflect.TypeOf((*ClusterStatusChangedEvent)(nil)).Elem()
}

func (b *ComputeResourceConfigInfo) GetComputeResourceConfigInfo() *ComputeResourceConfigInfo {
	return b
}

type BaseComputeResourceConfigInfo interface {
	GetComputeResourceConfigInfo() *ComputeResourceConfigInfo
}

func init() {
	t["BaseComputeResourceConfigInfo"] = reflect.TypeOf((*ComputeResourceConfigInfo)(nil)).Elem()
}

func (b *ComputeResourceConfigSpec) GetComputeResourceConfigSpec() *ComputeResourceConfigSpec {
	return b
}

type BaseComputeResourceConfigSpec interface {
	GetComputeResourceConfigSpec() *ComputeResourceConfigSpec
}

func init() {
	t["BaseComputeResourceConfigSpec"] = reflect.TypeOf((*ComputeResourceConfigSpec)(nil)).Elem()
}

func (b *ComputeResourceSummary) GetComputeResourceSummary() *ComputeResourceSummary { return b }

type BaseComputeResourceSummary interface {
	GetComputeResourceSummary() *ComputeResourceSummary
}

func init() {
	t["BaseComputeResourceSummary"] = reflect.TypeOf((*ComputeResourceSummary)(nil)).Elem()
}

func (b *CpuIncompatible) GetCpuIncompatible() *CpuIncompatible { return b }

type BaseCpuIncompatible interface {
	GetCpuIncompatible() *CpuIncompatible
}

func init() {
	t["BaseCpuIncompatible"] = reflect.TypeOf((*CpuIncompatible)(nil)).Elem()
}

func (b *CryptoSpec) GetCryptoSpec() *CryptoSpec { return b }

type BaseCryptoSpec interface {
	GetCryptoSpec() *CryptoSpec
}

func init() {
	t["BaseCryptoSpec"] = reflect.TypeOf((*CryptoSpec)(nil)).Elem()
}

func (b *CryptoSpecNoOp) GetCryptoSpecNoOp() *CryptoSpecNoOp { return b }

type BaseCryptoSpecNoOp interface {
	GetCryptoSpecNoOp() *CryptoSpecNoOp
}

func init() {
	t["BaseCryptoSpecNoOp"] = reflect.TypeOf((*CryptoSpecNoOp)(nil)).Elem()
}

func (b *CustomFieldDefEvent) GetCustomFieldDefEvent() *CustomFieldDefEvent { return b }

type BaseCustomFieldDefEvent interface {
	GetCustomFieldDefEvent() *CustomFieldDefEvent
}

func init() {
	t["BaseCustomFieldDefEvent"] = reflect.TypeOf((*CustomFieldDefEvent)(nil)).Elem()
}

func (b *CustomFieldEvent) GetCustomFieldEvent() *CustomFieldEvent { return b }

type BaseCustomFieldEvent interface {
	GetCustomFieldEvent() *CustomFieldEvent
}

func init() {
	t["BaseCustomFieldEvent"] = reflect.TypeOf((*CustomFieldEvent)(nil)).Elem()
}

func (b *CustomFieldValue) GetCustomFieldValue() *CustomFieldValue { return b }

type BaseCustomFieldValue interface {
	GetCustomFieldValue() *CustomFieldValue
}

func init() {
	t["BaseCustomFieldValue"] = reflect.TypeOf((*CustomFieldValue)(nil)).Elem()
}

func (b *CustomizationEvent) GetCustomizationEvent() *CustomizationEvent { return b }

type BaseCustomizationEvent interface {
	GetCustomizationEvent() *CustomizationEvent
}

func init() {
	t["BaseCustomizationEvent"] = reflect.TypeOf((*CustomizationEvent)(nil)).Elem()
}

func (b *CustomizationFailed) GetCustomizationFailed() *CustomizationFailed { return b }

type BaseCustomizationFailed interface {
	GetCustomizationFailed() *CustomizationFailed
}

func init() {
	t["BaseCustomizationFailed"] = reflect.TypeOf((*CustomizationFailed)(nil)).Elem()
}

func (b *CustomizationFault) GetCustomizationFault() *CustomizationFault { return b }

type BaseCustomizationFault interface {
	GetCustomizationFault() *CustomizationFault
}

func init() {
	t["BaseCustomizationFault"] = reflect.TypeOf((*CustomizationFault)(nil)).Elem()
}

func (b *CustomizationIdentitySettings) GetCustomizationIdentitySettings() *CustomizationIdentitySettings {
	return b
}

type BaseCustomizationIdentitySettings interface {
	GetCustomizationIdentitySettings() *CustomizationIdentitySettings
}

func init() {
	t["BaseCustomizationIdentitySettings"] = reflect.TypeOf((*CustomizationIdentitySettings)(nil)).Elem()
}

func (b *CustomizationIpGenerator) GetCustomizationIpGenerator() *CustomizationIpGenerator { return b }

type BaseCustomizationIpGenerator interface {
	GetCustomizationIpGenerator() *CustomizationIpGenerator
}

func init() {
	t["BaseCustomizationIpGenerator"] = reflect.TypeOf((*CustomizationIpGenerator)(nil)).Elem()
}

func (b *CustomizationIpV6Generator) GetCustomizationIpV6Generator() *CustomizationIpV6Generator {
	return b
}

type BaseCustomizationIpV6Generator interface {
	GetCustomizationIpV6Generator() *CustomizationIpV6Generator
}

func init() {
	t["BaseCustomizationIpV6Generator"] = reflect.TypeOf((*CustomizationIpV6Generator)(nil)).Elem()
}

func (b *CustomizationName) GetCustomizationName() *CustomizationName { return b }

type BaseCustomizationName interface {
	GetCustomizationName() *CustomizationName
}

func init() {
	t["BaseCustomizationName"] = reflect.TypeOf((*CustomizationName)(nil)).Elem()
}

func (b *CustomizationOptions) GetCustomizationOptions() *CustomizationOptions { return b }

type BaseCustomizationOptions interface {
	GetCustomizationOptions() *CustomizationOptions
}

func init() {
	t["BaseCustomizationOptions"] = reflect.TypeOf((*CustomizationOptions)(nil)).Elem()
}

func (b *DVPortSetting) GetDVPortSetting() *DVPortSetting { return b }

type BaseDVPortSetting interface {
	GetDVPortSetting() *DVPortSetting
}

func init() {
	t["BaseDVPortSetting"] = reflect.TypeOf((*DVPortSetting)(nil)).Elem()
}

func (b *DVPortgroupEvent) GetDVPortgroupEvent() *DVPortgroupEvent { return b }

type BaseDVPortgroupEvent interface {
	GetDVPortgroupEvent() *DVPortgroupEvent
}

func init() {
	t["BaseDVPortgroupEvent"] = reflect.TypeOf((*DVPortgroupEvent)(nil)).Elem()
}

func (b *DVPortgroupPolicy) GetDVPortgroupPolicy() *DVPortgroupPolicy { return b }

type BaseDVPortgroupPolicy interface {
	GetDVPortgroupPolicy() *DVPortgroupPolicy
}

func init() {
	t["BaseDVPortgroupPolicy"] = reflect.TypeOf((*DVPortgroupPolicy)(nil)).Elem()
}

func (b *DVSConfigInfo) GetDVSConfigInfo() *DVSConfigInfo { return b }

type BaseDVSConfigInfo interface {
	GetDVSConfigInfo() *DVSConfigInfo
}

func init() {
	t["BaseDVSConfigInfo"] = reflect.TypeOf((*DVSConfigInfo)(nil)).Elem()
}

func (b *DVSConfigSpec) GetDVSConfigSpec() *DVSConfigSpec { return b }

type BaseDVSConfigSpec interface {
	GetDVSConfigSpec() *DVSConfigSpec
}

func init() {
	t["BaseDVSConfigSpec"] = reflect.TypeOf((*DVSConfigSpec)(nil)).Elem()
}

func (b *DVSFeatureCapability) GetDVSFeatureCapability() *DVSFeatureCapability { return b }

type BaseDVSFeatureCapability interface {
	GetDVSFeatureCapability() *DVSFeatureCapability
}

func init() {
	t["BaseDVSFeatureCapability"] = reflect.TypeOf((*DVSFeatureCapability)(nil)).Elem()
}

func (b *DVSHealthCheckCapability) GetDVSHealthCheckCapability() *DVSHealthCheckCapability { return b }

type BaseDVSHealthCheckCapability interface {
	GetDVSHealthCheckCapability() *DVSHealthCheckCapability
}

func init() {
	t["BaseDVSHealthCheckCapability"] = reflect.TypeOf((*DVSHealthCheckCapability)(nil)).Elem()
}

func (b *DVSHealthCheckConfig) GetDVSHealthCheckConfig() *DVSHealthCheckConfig { return b }

type BaseDVSHealthCheckConfig interface {
	GetDVSHealthCheckConfig() *DVSHealthCheckConfig
}

func init() {
	t["BaseDVSHealthCheckConfig"] = reflect.TypeOf((*DVSHealthCheckConfig)(nil)).Elem()
}

func (b *DVSUplinkPortPolicy) GetDVSUplinkPortPolicy() *DVSUplinkPortPolicy { return b }

type BaseDVSUplinkPortPolicy interface {
	GetDVSUplinkPortPolicy() *DVSUplinkPortPolicy
}

func init() {
	t["BaseDVSUplinkPortPolicy"] = reflect.TypeOf((*DVSUplinkPortPolicy)(nil)).Elem()
}

func (b *DailyTaskScheduler) GetDailyTaskScheduler() *DailyTaskScheduler { return b }

type BaseDailyTaskScheduler interface {
	GetDailyTaskScheduler() *DailyTaskScheduler
}

func init() {
	t["BaseDailyTaskScheduler"] = reflect.TypeOf((*DailyTaskScheduler)(nil)).Elem()
}

func (b *DatacenterEvent) GetDatacenterEvent() *DatacenterEvent { return b }

type BaseDatacenterEvent interface {
	GetDatacenterEvent() *DatacenterEvent
}

func init() {
	t["BaseDatacenterEvent"] = reflect.TypeOf((*DatacenterEvent)(nil)).Elem()
}

func (b *DatastoreEvent) GetDatastoreEvent() *DatastoreEvent { return b }

type BaseDatastoreEvent interface {
	GetDatastoreEvent() *DatastoreEvent
}

func init() {
	t["BaseDatastoreEvent"] = reflect.TypeOf((*DatastoreEvent)(nil)).Elem()
}

func (b *DatastoreFileEvent) GetDatastoreFileEvent() *DatastoreFileEvent { return b }

type BaseDatastoreFileEvent interface {
	GetDatastoreFileEvent() *DatastoreFileEvent
}

func init() {
	t["BaseDatastoreFileEvent"] = reflect.TypeOf((*DatastoreFileEvent)(nil)).Elem()
}

func (b *DatastoreInfo) GetDatastoreInfo() *DatastoreInfo { return b }

type BaseDatastoreInfo interface {
	GetDatastoreInfo() *DatastoreInfo
}

func init() {
	t["BaseDatastoreInfo"] = reflect.TypeOf((*DatastoreInfo)(nil)).Elem()
}

func (b *DatastoreNotWritableOnHost) GetDatastoreNotWritableOnHost() *DatastoreNotWritableOnHost {
	return b
}

type BaseDatastoreNotWritableOnHost interface {
	GetDatastoreNotWritableOnHost() *DatastoreNotWritableOnHost
}

func init() {
	t["BaseDatastoreNotWritableOnHost"] = reflect.TypeOf((*DatastoreNotWritableOnHost)(nil)).Elem()
}

func (b *Description) GetDescription() *Description { return b }

type BaseDescription interface {
	GetDescription() *Description
}

func init() {
	t["BaseDescription"] = reflect.TypeOf((*Description)(nil)).Elem()
}

func (b *DeviceBackingNotSupported) GetDeviceBackingNotSupported() *DeviceBackingNotSupported {
	return b
}

type BaseDeviceBackingNotSupported interface {
	GetDeviceBackingNotSupported() *DeviceBackingNotSupported
}

func init() {
	t["BaseDeviceBackingNotSupported"] = reflect.TypeOf((*DeviceBackingNotSupported)(nil)).Elem()
}

func (b *DeviceNotSupported) GetDeviceNotSupported() *DeviceNotSupported { return b }

type BaseDeviceNotSupported interface {
	GetDeviceNotSupported() *DeviceNotSupported
}

func init() {
	t["BaseDeviceNotSupported"] = reflect.TypeOf((*DeviceNotSupported)(nil)).Elem()
}

func (b *DiskNotSupported) GetDiskNotSupported() *DiskNotSupported { return b }

type BaseDiskNotSupported interface {
	GetDiskNotSupported() *DiskNotSupported
}

func init() {
	t["BaseDiskNotSupported"] = reflect.TypeOf((*DiskNotSupported)(nil)).Elem()
}

func (b *DistributedVirtualSwitchHostMemberBacking) GetDistributedVirtualSwitchHostMemberBacking() *DistributedVirtualSwitchHostMemberBacking {
	return b
}

type BaseDistributedVirtualSwitchHostMemberBacking interface {
	GetDistributedVirtualSwitchHostMemberBacking() *DistributedVirtualSwitchHostMemberBacking
}

func init() {
	t["BaseDistributedVirtualSwitchHostMemberBacking"] = reflect.TypeOf((*DistributedVirtualSwitchHostMemberBacking)(nil)).Elem()
}

func (b *DistributedVirtualSwitchManagerHostDvsFilterSpec) GetDistributedVirtualSwitchManagerHostDvsFilterSpec() *DistributedVirtualSwitchManagerHostDvsFilterSpec {
	return b
}

type BaseDistributedVirtualSwitchManagerHostDvsFilterSpec interface {
	GetDistributedVirtualSwitchManagerHostDvsFilterSpec() *DistributedVirtualSwitchManagerHostDvsFilterSpec
}

func init() {
	t["BaseDistributedVirtualSwitchManagerHostDvsFilterSpec"] = reflect.TypeOf((*DistributedVirtualSwitchManagerHostDvsFilterSpec)(nil)).Elem()
}

func (b *DvsEvent) GetDvsEvent() *DvsEvent { return b }

type BaseDvsEvent interface {
	GetDvsEvent() *DvsEvent
}

func init() {
	t["BaseDvsEvent"] = reflect.TypeOf((*DvsEvent)(nil)).Elem()
}

func (b *DvsFault) GetDvsFault() *DvsFault { return b }

type BaseDvsFault interface {
	GetDvsFault() *DvsFault
}

func init() {
	t["BaseDvsFault"] = reflect.TypeOf((*DvsFault)(nil)).Elem()
}

func (b *DvsFilterConfig) GetDvsFilterConfig() *DvsFilterConfig { return b }

type BaseDvsFilterConfig interface {
	GetDvsFilterConfig() *DvsFilterConfig
	GetDvsTrafficFilterConfig() *DvsTrafficFilterConfig
}

func init() {
	t["BaseDvsFilterConfig"] = reflect.TypeOf((*DvsFilterConfig)(nil)).Elem()
}

func (b *DvsHealthStatusChangeEvent) GetDvsHealthStatusChangeEvent() *DvsHealthStatusChangeEvent {
	return b
}

type BaseDvsHealthStatusChangeEvent interface {
	GetDvsHealthStatusChangeEvent() *DvsHealthStatusChangeEvent
}

func init() {
	t["BaseDvsHealthStatusChangeEvent"] = reflect.TypeOf((*DvsHealthStatusChangeEvent)(nil)).Elem()
}

func (b *DvsIpPort) GetDvsIpPort() *DvsIpPort { return b }

type BaseDvsIpPort interface {
	GetDvsIpPort() *DvsIpPort
}

func init() {
	t["BaseDvsIpPort"] = reflect.TypeOf((*DvsIpPort)(nil)).Elem()
}

func (b *DvsNetworkRuleAction) GetDvsNetworkRuleAction() *DvsNetworkRuleAction { return b }

type BaseDvsNetworkRuleAction interface {
	GetDvsNetworkRuleAction() *DvsNetworkRuleAction
}

func init() {
	t["BaseDvsNetworkRuleAction"] = reflect.TypeOf((*DvsNetworkRuleAction)(nil)).Elem()
}

func (b *DvsNetworkRuleQualifier) GetDvsNetworkRuleQualifier() *DvsNetworkRuleQualifier { return b }

type BaseDvsNetworkRuleQualifier interface {
	GetDvsNetworkRuleQualifier() *DvsNetworkRuleQualifier
	GetDvsIpNetworkRuleQualifier() *DvsIpNetworkRuleQualifier
}

func init() {
	t["BaseDvsNetworkRuleQualifier"] = reflect.TypeOf((*DvsNetworkRuleQualifier)(nil)).Elem()
}

func (b *DvsIpNetworkRuleQualifier) GetDvsIpNetworkRuleQualifier() *DvsIpNetworkRuleQualifier {
	return b
}

type BaseDvsIpNetworkRuleQualifier interface {
	GetDvsIpNetworkRuleQualifier() *DvsIpNetworkRuleQualifier
}

func (b *DvsTrafficFilterConfig) GetDvsTrafficFilterConfig() *DvsTrafficFilterConfig { return b }

type BaseDvsTrafficFilterConfig interface {
	GetDvsTrafficFilterConfig() *DvsTrafficFilterConfig
}

func init() {
	t["BaseDvsTrafficFilterConfig"] = reflect.TypeOf((*DvsTrafficFilterConfig)(nil)).Elem()
}

func (b *DvsVNicProfile) GetDvsVNicProfile() *DvsVNicProfile { return b }

type BaseDvsVNicProfile interface {
	GetDvsVNicProfile() *DvsVNicProfile
}

func init() {
	t["BaseDvsVNicProfile"] = reflect.TypeOf((*DvsVNicProfile)(nil)).Elem()
}

func (b *DynamicData) GetDynamicData() *DynamicData { return b }

type BaseDynamicData interface {
	GetDynamicData() *DynamicData
}

func init() {
	t["BaseDynamicData"] = reflect.TypeOf((*DynamicData)(nil)).Elem()
}

func (b *EVCAdmissionFailed) GetEVCAdmissionFailed() *EVCAdmissionFailed { return b }

type BaseEVCAdmissionFailed interface {
	GetEVCAdmissionFailed() *EVCAdmissionFailed
}

func init() {
	t["BaseEVCAdmissionFailed"] = reflect.TypeOf((*EVCAdmissionFailed)(nil)).Elem()
}

func (b *EVCConfigFault) GetEVCConfigFault() *EVCConfigFault { return b }

type BaseEVCConfigFault interface {
	GetEVCConfigFault() *EVCConfigFault
}

func init() {
	t["BaseEVCConfigFault"] = reflect.TypeOf((*EVCConfigFault)(nil)).Elem()
}

func (b *ElementDescription) GetElementDescription() *ElementDescription { return b }

type BaseElementDescription interface {
	GetElementDescription() *ElementDescription
}

func init() {
	t["BaseElementDescription"] = reflect.TypeOf((*ElementDescription)(nil)).Elem()
}

func (b *EnteredStandbyModeEvent) GetEnteredStandbyModeEvent() *EnteredStandbyModeEvent { return b }

type BaseEnteredStandbyModeEvent interface {
	GetEnteredStandbyModeEvent() *EnteredStandbyModeEvent
}

func init() {
	t["BaseEnteredStandbyModeEvent"] = reflect.TypeOf((*EnteredStandbyModeEvent)(nil)).Elem()
}

func (b *EnteringStandbyModeEvent) GetEnteringStandbyModeEvent() *EnteringStandbyModeEvent { return b }

type BaseEnteringStandbyModeEvent interface {
	GetEnteringStandbyModeEvent() *EnteringStandbyModeEvent
}

func init() {
	t["BaseEnteringStandbyModeEvent"] = reflect.TypeOf((*EnteringStandbyModeEvent)(nil)).Elem()
}

func (b *EntityEventArgument) GetEntityEventArgument() *EntityEventArgument { return b }

type BaseEntityEventArgument interface {
	GetEntityEventArgument() *EntityEventArgument
}

func init() {
	t["BaseEntityEventArgument"] = reflect.TypeOf((*EntityEventArgument)(nil)).Elem()
}

func (b *Event) GetEvent() *Event { return b }

type BaseEvent interface {
	GetEvent() *Event
}

func init() {
	t["BaseEvent"] = reflect.TypeOf((*Event)(nil)).Elem()
}

func (b *EventArgument) GetEventArgument() *EventArgument { return b }

type BaseEventArgument interface {
	GetEventArgument() *EventArgument
}

func init() {
	t["BaseEventArgument"] = reflect.TypeOf((*EventArgument)(nil)).Elem()
}

func (b *ExitStandbyModeFailedEvent) GetExitStandbyModeFailedEvent() *ExitStandbyModeFailedEvent {
	return b
}

type BaseExitStandbyModeFailedEvent interface {
	GetExitStandbyModeFailedEvent() *ExitStandbyModeFailedEvent
}

func init() {
	t["BaseExitStandbyModeFailedEvent"] = reflect.TypeOf((*ExitStandbyModeFailedEvent)(nil)).Elem()
}

func (b *ExitedStandbyModeEvent) GetExitedStandbyModeEvent() *ExitedStandbyModeEvent { return b }

type BaseExitedStandbyModeEvent interface {
	GetExitedStandbyModeEvent() *ExitedStandbyModeEvent
}

func init() {
	t["BaseExitedStandbyModeEvent"] = reflect.TypeOf((*ExitedStandbyModeEvent)(nil)).Elem()
}

func (b *ExitingStandbyModeEvent) GetExitingStandbyModeEvent() *ExitingStandbyModeEvent { return b }

type BaseExitingStandbyModeEvent interface {
	GetExitingStandbyModeEvent() *ExitingStandbyModeEvent
}

func init() {
	t["BaseExitingStandbyModeEvent"] = reflect.TypeOf((*ExitingStandbyModeEvent)(nil)).Elem()
}

func (b *ExpiredFeatureLicense) GetExpiredFeatureLicense() *ExpiredFeatureLicense { return b }

type BaseExpiredFeatureLicense interface {
	GetExpiredFeatureLicense() *ExpiredFeatureLicense
}

func init() {
	t["BaseExpiredFeatureLicense"] = reflect.TypeOf((*ExpiredFeatureLicense)(nil)).Elem()
}

func (b *FaultToleranceConfigInfo) GetFaultToleranceConfigInfo() *FaultToleranceConfigInfo { return b }

type BaseFaultToleranceConfigInfo interface {
	GetFaultToleranceConfigInfo() *FaultToleranceConfigInfo
}

func init() {
	t["BaseFaultToleranceConfigInfo"] = reflect.TypeOf((*FaultToleranceConfigInfo)(nil)).Elem()
}

func (b *FcoeFault) GetFcoeFault() *FcoeFault { return b }

type BaseFcoeFault interface {
	GetFcoeFault() *FcoeFault
}

func init() {
	t["BaseFcoeFault"] = reflect.TypeOf((*FcoeFault)(nil)).Elem()
}

func (b *FileBackedVirtualDiskSpec) GetFileBackedVirtualDiskSpec() *FileBackedVirtualDiskSpec {
	return b
}

type BaseFileBackedVirtualDiskSpec interface {
	GetFileBackedVirtualDiskSpec() *FileBackedVirtualDiskSpec
}

func init() {
	t["BaseFileBackedVirtualDiskSpec"] = reflect.TypeOf((*FileBackedVirtualDiskSpec)(nil)).Elem()
}

func (b *FileFault) GetFileFault() *FileFault { return b }

type BaseFileFault interface {
	GetFileFault() *FileFault
}

func init() {
	t["BaseFileFault"] = reflect.TypeOf((*FileFault)(nil)).Elem()
}

func (b *FileInfo) GetFileInfo() *FileInfo { return b }

type BaseFileInfo interface {
	GetFileInfo() *FileInfo
}

func init() {
	t["BaseFileInfo"] = reflect.TypeOf((*FileInfo)(nil)).Elem()
}

func (b *FileQuery) GetFileQuery() *FileQuery { return b }

type BaseFileQuery interface {
	GetFileQuery() *FileQuery
}

func init() {
	t["BaseFileQuery"] = reflect.TypeOf((*FileQuery)(nil)).Elem()
}

func (b *GatewayConnectFault) GetGatewayConnectFault() *GatewayConnectFault { return b }

type BaseGatewayConnectFault interface {
	GetGatewayConnectFault() *GatewayConnectFault
}

func init() {
	t["BaseGatewayConnectFault"] = reflect.TypeOf((*GatewayConnectFault)(nil)).Elem()
}

func (b *GatewayToHostConnectFault) GetGatewayToHostConnectFault() *GatewayToHostConnectFault {
	return b
}

type BaseGatewayToHostConnectFault interface {
	GetGatewayToHostConnectFault() *GatewayToHostConnectFault
}

func init() {
	t["BaseGatewayToHostConnectFault"] = reflect.TypeOf((*GatewayToHostConnectFault)(nil)).Elem()
}

func (b *GeneralEvent) GetGeneralEvent() *GeneralEvent { return b }

type BaseGeneralEvent interface {
	GetGeneralEvent() *GeneralEvent
}

func init() {
	t["BaseGeneralEvent"] = reflect.TypeOf((*GeneralEvent)(nil)).Elem()
}

func (b *GuestAuthSubject) GetGuestAuthSubject() *GuestAuthSubject { return b }

type BaseGuestAuthSubject interface {
	GetGuestAuthSubject() *GuestAuthSubject
}

func init() {
	t["BaseGuestAuthSubject"] = reflect.TypeOf((*GuestAuthSubject)(nil)).Elem()
}

func (b *GuestAuthentication) GetGuestAuthentication() *GuestAuthentication { return b }

type BaseGuestAuthentication interface {
	GetGuestAuthentication() *GuestAuthentication
}

func init() {
	t["BaseGuestAuthentication"] = reflect.TypeOf((*GuestAuthentication)(nil)).Elem()
}

func (b *GuestFileAttributes) GetGuestFileAttributes() *GuestFileAttributes { return b }

type BaseGuestFileAttributes interface {
	GetGuestFileAttributes() *GuestFileAttributes
}

func init() {
	t["BaseGuestFileAttributes"] = reflect.TypeOf((*GuestFileAttributes)(nil)).Elem()
}

func (b *GuestOperationsFault) GetGuestOperationsFault() *GuestOperationsFault { return b }

type BaseGuestOperationsFault interface {
	GetGuestOperationsFault() *GuestOperationsFault
}

func init() {
	t["BaseGuestOperationsFault"] = reflect.TypeOf((*GuestOperationsFault)(nil)).Elem()
}

func (b *GuestProgramSpec) GetGuestProgramSpec() *GuestProgramSpec { return b }

type BaseGuestProgramSpec interface {
	GetGuestProgramSpec() *GuestProgramSpec
}

func init() {
	t["BaseGuestProgramSpec"] = reflect.TypeOf((*GuestProgramSpec)(nil)).Elem()
}

func (b *GuestRegValueDataSpec) GetGuestRegValueDataSpec() *GuestRegValueDataSpec { return b }

type BaseGuestRegValueDataSpec interface {
	GetGuestRegValueDataSpec() *GuestRegValueDataSpec
}

func init() {
	t["BaseGuestRegValueDataSpec"] = reflect.TypeOf((*GuestRegValueDataSpec)(nil)).Elem()
}

func (b *GuestRegistryFault) GetGuestRegistryFault() *GuestRegistryFault { return b }

type BaseGuestRegistryFault interface {
	GetGuestRegistryFault() *GuestRegistryFault
}

func init() {
	t["BaseGuestRegistryFault"] = reflect.TypeOf((*GuestRegistryFault)(nil)).Elem()
}

func (b *GuestRegistryKeyFault) GetGuestRegistryKeyFault() *GuestRegistryKeyFault { return b }

type BaseGuestRegistryKeyFault interface {
	GetGuestRegistryKeyFault() *GuestRegistryKeyFault
}

func init() {
	t["BaseGuestRegistryKeyFault"] = reflect.TypeOf((*GuestRegistryKeyFault)(nil)).Elem()
}

func (b *GuestRegistryValueFault) GetGuestRegistryValueFault() *GuestRegistryValueFault { return b }

type BaseGuestRegistryValueFault interface {
	GetGuestRegistryValueFault() *GuestRegistryValueFault
}

func init() {
	t["BaseGuestRegistryValueFault"] = reflect.TypeOf((*GuestRegistryValueFault)(nil)).Elem()
}

func (b *HostAccountSpec) GetHostAccountSpec() *HostAccountSpec { return b }

type BaseHostAccountSpec interface {
	GetHostAccountSpec() *HostAccountSpec
}

func init() {
	t["BaseHostAccountSpec"] = reflect.TypeOf((*HostAccountSpec)(nil)).Elem()
}

func (b *HostAuthenticationStoreInfo) GetHostAuthenticationStoreInfo() *HostAuthenticationStoreInfo {
	return b
}

type BaseHostAuthenticationStoreInfo interface {
	GetHostAuthenticationStoreInfo() *HostAuthenticationStoreInfo
}

func init() {
	t["BaseHostAuthenticationStoreInfo"] = reflect.TypeOf((*HostAuthenticationStoreInfo)(nil)).Elem()
}

func (b *HostCommunication) GetHostCommunication() *HostCommunication { return b }

type BaseHostCommunication interface {
	GetHostCommunication() *HostCommunication
}

func init() {
	t["BaseHostCommunication"] = reflect.TypeOf((*HostCommunication)(nil)).Elem()
}

func (b *HostConfigFault) GetHostConfigFault() *HostConfigFault { return b }

type BaseHostConfigFault interface {
	GetHostConfigFault() *HostConfigFault
}

func init() {
	t["BaseHostConfigFault"] = reflect.TypeOf((*HostConfigFault)(nil)).Elem()
}

func (b *HostConnectFault) GetHostConnectFault() *HostConnectFault { return b }

type BaseHostConnectFault interface {
	GetHostConnectFault() *HostConnectFault
}

func init() {
	t["BaseHostConnectFault"] = reflect.TypeOf((*HostConnectFault)(nil)).Elem()
}

func (b *HostConnectInfoNetworkInfo) GetHostConnectInfoNetworkInfo() *HostConnectInfoNetworkInfo {
	return b
}

type BaseHostConnectInfoNetworkInfo interface {
	GetHostConnectInfoNetworkInfo() *HostConnectInfoNetworkInfo
}

func init() {
	t["BaseHostConnectInfoNetworkInfo"] = reflect.TypeOf((*HostConnectInfoNetworkInfo)(nil)).Elem()
}

func (b *HostDasEvent) GetHostDasEvent() *HostDasEvent { return b }

type BaseHostDasEvent interface {
	GetHostDasEvent() *HostDasEvent
}

func init() {
	t["BaseHostDasEvent"] = reflect.TypeOf((*HostDasEvent)(nil)).Elem()
}

func (b *HostDatastoreConnectInfo) GetHostDatastoreConnectInfo() *HostDatastoreConnectInfo { return b }

type BaseHostDatastoreConnectInfo interface {
	GetHostDatastoreConnectInfo() *HostDatastoreConnectInfo
}

func init() {
	t["BaseHostDatastoreConnectInfo"] = reflect.TypeOf((*HostDatastoreConnectInfo)(nil)).Elem()
}

func (b *HostDevice) GetHostDevice() *HostDevice { return b }

type BaseHostDevice interface {
	GetHostDevice() *HostDevice
}

func init() {
	t["BaseHostDevice"] = reflect.TypeOf((*HostDevice)(nil)).Elem()
}

func (b *HostDigestInfo) GetHostDigestInfo() *HostDigestInfo { return b }

type BaseHostDigestInfo interface {
	GetHostDigestInfo() *HostDigestInfo
}

func init() {
	t["BaseHostDigestInfo"] = reflect.TypeOf((*HostDigestInfo)(nil)).Elem()
}

func (b *HostDirectoryStoreInfo) GetHostDirectoryStoreInfo() *HostDirectoryStoreInfo { return b }

type BaseHostDirectoryStoreInfo interface {
	GetHostDirectoryStoreInfo() *HostDirectoryStoreInfo
}

func init() {
	t["BaseHostDirectoryStoreInfo"] = reflect.TypeOf((*HostDirectoryStoreInfo)(nil)).Elem()
}

func (b *HostDnsConfig) GetHostDnsConfig() *HostDnsConfig { return b }

type BaseHostDnsConfig interface {
	GetHostDnsConfig() *HostDnsConfig
}

func init() {
	t["BaseHostDnsConfig"] = reflect.TypeOf((*HostDnsConfig)(nil)).Elem()
}

func (b *HostEvent) GetHostEvent() *HostEvent { return b }

type BaseHostEvent interface {
	GetHostEvent() *HostEvent
}

func init() {
	t["BaseHostEvent"] = reflect.TypeOf((*HostEvent)(nil)).Elem()
}

func (b *HostFibreChannelHba) GetHostFibreChannelHba() *HostFibreChannelHba { return b }

type BaseHostFibreChannelHba interface {
	GetHostFibreChannelHba() *HostFibreChannelHba
}

func init() {
	t["BaseHostFibreChannelHba"] = reflect.TypeOf((*HostFibreChannelHba)(nil)).Elem()
}

func (b *HostFibreChannelTargetTransport) GetHostFibreChannelTargetTransport() *HostFibreChannelTargetTransport {
	return b
}

type BaseHostFibreChannelTargetTransport interface {
	GetHostFibreChannelTargetTransport() *HostFibreChannelTargetTransport
}

func init() {
	t["BaseHostFibreChannelTargetTransport"] = reflect.TypeOf((*HostFibreChannelTargetTransport)(nil)).Elem()
}

func (b *HostFileSystemVolume) GetHostFileSystemVolume() *HostFileSystemVolume { return b }

type BaseHostFileSystemVolume interface {
	GetHostFileSystemVolume() *HostFileSystemVolume
}

func init() {
	t["BaseHostFileSystemVolume"] = reflect.TypeOf((*HostFileSystemVolume)(nil)).Elem()
}

func (b *HostHardwareElementInfo) GetHostHardwareElementInfo() *HostHardwareElementInfo { return b }

type BaseHostHardwareElementInfo interface {
	GetHostHardwareElementInfo() *HostHardwareElementInfo
}

func init() {
	t["BaseHostHardwareElementInfo"] = reflect.TypeOf((*HostHardwareElementInfo)(nil)).Elem()
}

func (b *HostHostBusAdapter) GetHostHostBusAdapter() *HostHostBusAdapter { return b }

type BaseHostHostBusAdapter interface {
	GetHostHostBusAdapter() *HostHostBusAdapter
}

func init() {
	t["BaseHostHostBusAdapter"] = reflect.TypeOf((*HostHostBusAdapter)(nil)).Elem()
}

func (b *HostIpRouteConfig) GetHostIpRouteConfig() *HostIpRouteConfig { return b }

type BaseHostIpRouteConfig interface {
	GetHostIpRouteConfig() *HostIpRouteConfig
}

func init() {
	t["BaseHostIpRouteConfig"] = reflect.TypeOf((*HostIpRouteConfig)(nil)).Elem()
}

func (b *HostMemberHealthCheckResult) GetHostMemberHealthCheckResult() *HostMemberHealthCheckResult {
	return b
}

type BaseHostMemberHealthCheckResult interface {
	GetHostMemberHealthCheckResult() *HostMemberHealthCheckResult
}

func init() {
	t["BaseHostMemberHealthCheckResult"] = reflect.TypeOf((*HostMemberHealthCheckResult)(nil)).Elem()
}

func (b *HostMemberUplinkHealthCheckResult) GetHostMemberUplinkHealthCheckResult() *HostMemberUplinkHealthCheckResult {
	return b
}

type BaseHostMemberUplinkHealthCheckResult interface {
	GetHostMemberUplinkHealthCheckResult() *HostMemberUplinkHealthCheckResult
}

func init() {
	t["BaseHostMemberUplinkHealthCheckResult"] = reflect.TypeOf((*HostMemberUplinkHealthCheckResult)(nil)).Elem()
}

func (b *HostMultipathInfoLogicalUnitPolicy) GetHostMultipathInfoLogicalUnitPolicy() *HostMultipathInfoLogicalUnitPolicy {
	return b
}

type BaseHostMultipathInfoLogicalUnitPolicy interface {
	GetHostMultipathInfoLogicalUnitPolicy() *HostMultipathInfoLogicalUnitPolicy
}

func init() {
	t["BaseHostMultipathInfoLogicalUnitPolicy"] = reflect.TypeOf((*HostMultipathInfoLogicalUnitPolicy)(nil)).Elem()
}

func (b *HostPciPassthruConfig) GetHostPciPassthruConfig() *HostPciPassthruConfig { return b }

type BaseHostPciPassthruConfig interface {
	GetHostPciPassthruConfig() *HostPciPassthruConfig
}

func init() {
	t["BaseHostPciPassthruConfig"] = reflect.TypeOf((*HostPciPassthruConfig)(nil)).Elem()
}

func (b *HostPciPassthruInfo) GetHostPciPassthruInfo() *HostPciPassthruInfo { return b }

type BaseHostPciPassthruInfo interface {
	GetHostPciPassthruInfo() *HostPciPassthruInfo
}

func init() {
	t["BaseHostPciPassthruInfo"] = reflect.TypeOf((*HostPciPassthruInfo)(nil)).Elem()
}

func (b *HostPowerOpFailed) GetHostPowerOpFailed() *HostPowerOpFailed { return b }

type BaseHostPowerOpFailed interface {
	GetHostPowerOpFailed() *HostPowerOpFailed
}

func init() {
	t["BaseHostPowerOpFailed"] = reflect.TypeOf((*HostPowerOpFailed)(nil)).Elem()
}

func (b *HostProfileConfigSpec) GetHostProfileConfigSpec() *HostProfileConfigSpec { return b }

type BaseHostProfileConfigSpec interface {
	GetHostProfileConfigSpec() *HostProfileConfigSpec
}

func init() {
	t["BaseHostProfileConfigSpec"] = reflect.TypeOf((*HostProfileConfigSpec)(nil)).Elem()
}

func (b *HostProfilesEntityCustomizations) GetHostProfilesEntityCustomizations() *HostProfilesEntityCustomizations {
	return b
}

type BaseHostProfilesEntityCustomizations interface {
	GetHostProfilesEntityCustomizations() *HostProfilesEntityCustomizations
}

func init() {
	t["BaseHostProfilesEntityCustomizations"] = reflect.TypeOf((*HostProfilesEntityCustomizations)(nil)).Elem()
}

func (b *HostSriovDevicePoolInfo) GetHostSriovDevicePoolInfo() *HostSriovDevicePoolInfo { return b }

type BaseHostSriovDevicePoolInfo interface {
	GetHostSriovDevicePoolInfo() *HostSriovDevicePoolInfo
}

func init() {
	t["BaseHostSriovDevicePoolInfo"] = reflect.TypeOf((*HostSriovDevicePoolInfo)(nil)).Elem()
}

func (b *HostSystemSwapConfigurationSystemSwapOption) GetHostSystemSwapConfigurationSystemSwapOption() *HostSystemSwapConfigurationSystemSwapOption {
	return b
}

type BaseHostSystemSwapConfigurationSystemSwapOption interface {
	GetHostSystemSwapConfigurationSystemSwapOption() *HostSystemSwapConfigurationSystemSwapOption
}

func init() {
	t["BaseHostSystemSwapConfigurationSystemSwapOption"] = reflect.TypeOf((*HostSystemSwapConfigurationSystemSwapOption)(nil)).Elem()
}

func (b *HostTargetTransport) GetHostTargetTransport() *HostTargetTransport { return b }

type BaseHostTargetTransport interface {
	GetHostTargetTransport() *HostTargetTransport
}

func init() {
	t["BaseHostTargetTransport"] = reflect.TypeOf((*HostTargetTransport)(nil)).Elem()
}

func (b *HostTpmEventDetails) GetHostTpmEventDetails() *HostTpmEventDetails { return b }

type BaseHostTpmEventDetails interface {
	GetHostTpmEventDetails() *HostTpmEventDetails
}

func init() {
	t["BaseHostTpmEventDetails"] = reflect.TypeOf((*HostTpmEventDetails)(nil)).Elem()
}

func (b *HostVirtualSwitchBridge) GetHostVirtualSwitchBridge() *HostVirtualSwitchBridge { return b }

type BaseHostVirtualSwitchBridge interface {
	GetHostVirtualSwitchBridge() *HostVirtualSwitchBridge
}

func init() {
	t["BaseHostVirtualSwitchBridge"] = reflect.TypeOf((*HostVirtualSwitchBridge)(nil)).Elem()
}

func (b *HourlyTaskScheduler) GetHourlyTaskScheduler() *HourlyTaskScheduler { return b }

type BaseHourlyTaskScheduler interface {
	GetHourlyTaskScheduler() *HourlyTaskScheduler
}

func init() {
	t["BaseHourlyTaskScheduler"] = reflect.TypeOf((*HourlyTaskScheduler)(nil)).Elem()
}

func (b *ImportSpec) GetImportSpec() *ImportSpec { return b }

type BaseImportSpec interface {
	GetImportSpec() *ImportSpec
}

func init() {
	t["BaseImportSpec"] = reflect.TypeOf((*ImportSpec)(nil)).Elem()
}

func (b *InaccessibleDatastore) GetInaccessibleDatastore() *InaccessibleDatastore { return b }

type BaseInaccessibleDatastore interface {
	GetInaccessibleDatastore() *InaccessibleDatastore
}

func init() {
	t["BaseInaccessibleDatastore"] = reflect.TypeOf((*InaccessibleDatastore)(nil)).Elem()
}

func (b *InheritablePolicy) GetInheritablePolicy() *InheritablePolicy { return b }

type BaseInheritablePolicy interface {
	GetInheritablePolicy() *InheritablePolicy
}

func init() {
	t["BaseInheritablePolicy"] = reflect.TypeOf((*InheritablePolicy)(nil)).Elem()
}

func (b *InsufficientHostCapacityFault) GetInsufficientHostCapacityFault() *InsufficientHostCapacityFault {
	return b
}

type BaseInsufficientHostCapacityFault interface {
	GetInsufficientHostCapacityFault() *InsufficientHostCapacityFault
}

func init() {
	t["BaseInsufficientHostCapacityFault"] = reflect.TypeOf((*InsufficientHostCapacityFault)(nil)).Elem()
}

func (b *InsufficientResourcesFault) GetInsufficientResourcesFault() *InsufficientResourcesFault {
	return b
}

type BaseInsufficientResourcesFault interface {
	GetInsufficientResourcesFault() *InsufficientResourcesFault
}

func init() {
	t["BaseInsufficientResourcesFault"] = reflect.TypeOf((*InsufficientResourcesFault)(nil)).Elem()
}

func (b *InsufficientStandbyResource) GetInsufficientStandbyResource() *InsufficientStandbyResource {
	return b
}

type BaseInsufficientStandbyResource interface {
	GetInsufficientStandbyResource() *InsufficientStandbyResource
}

func init() {
	t["BaseInsufficientStandbyResource"] = reflect.TypeOf((*InsufficientStandbyResource)(nil)).Elem()
}

func (b *InvalidArgument) GetInvalidArgument() *InvalidArgument { return b }

type BaseInvalidArgument interface {
	GetInvalidArgument() *InvalidArgument
}

func init() {
	t["BaseInvalidArgument"] = reflect.TypeOf((*InvalidArgument)(nil)).Elem()
}

func (b *InvalidCAMServer) GetInvalidCAMServer() *InvalidCAMServer { return b }

type BaseInvalidCAMServer interface {
	GetInvalidCAMServer() *InvalidCAMServer
}

func init() {
	t["BaseInvalidCAMServer"] = reflect.TypeOf((*InvalidCAMServer)(nil)).Elem()
}

func (b *InvalidDatastore) GetInvalidDatastore() *InvalidDatastore { return b }

type BaseInvalidDatastore interface {
	GetInvalidDatastore() *InvalidDatastore
}

func init() {
	t["BaseInvalidDatastore"] = reflect.TypeOf((*InvalidDatastore)(nil)).Elem()
}

func (b *InvalidDeviceSpec) GetInvalidDeviceSpec() *InvalidDeviceSpec { return b }

type BaseInvalidDeviceSpec interface {
	GetInvalidDeviceSpec() *InvalidDeviceSpec
}

func init() {
	t["BaseInvalidDeviceSpec"] = reflect.TypeOf((*InvalidDeviceSpec)(nil)).Elem()
}

func (b *InvalidFolder) GetInvalidFolder() *InvalidFolder { return b }

type BaseInvalidFolder interface {
	GetInvalidFolder() *InvalidFolder
}

func init() {
	t["BaseInvalidFolder"] = reflect.TypeOf((*InvalidFolder)(nil)).Elem()
}

func (b *InvalidFormat) GetInvalidFormat() *InvalidFormat { return b }

type BaseInvalidFormat interface {
	GetInvalidFormat() *InvalidFormat
}

func init() {
	t["BaseInvalidFormat"] = reflect.TypeOf((*InvalidFormat)(nil)).Elem()
}

func (b *InvalidHostState) GetInvalidHostState() *InvalidHostState { return b }

type BaseInvalidHostState interface {
	GetInvalidHostState() *InvalidHostState
}

func init() {
	t["BaseInvalidHostState"] = reflect.TypeOf((*InvalidHostState)(nil)).Elem()
}

func (b *InvalidLogin) GetInvalidLogin() *InvalidLogin { return b }

type BaseInvalidLogin interface {
	GetInvalidLogin() *InvalidLogin
}

func init() {
	t["BaseInvalidLogin"] = reflect.TypeOf((*InvalidLogin)(nil)).Elem()
}

func (b *InvalidPropertyValue) GetInvalidPropertyValue() *InvalidPropertyValue { return b }

type BaseInvalidPropertyValue interface {
	GetInvalidPropertyValue() *InvalidPropertyValue
}

func init() {
	t["BaseInvalidPropertyValue"] = reflect.TypeOf((*InvalidPropertyValue)(nil)).Elem()
}

func (b *InvalidRequest) GetInvalidRequest() *InvalidRequest { return b }

type BaseInvalidRequest interface {
	GetInvalidRequest() *InvalidRequest
}

func init() {
	t["BaseInvalidRequest"] = reflect.TypeOf((*InvalidRequest)(nil)).Elem()
}

func (b *InvalidState) GetInvalidState() *InvalidState { return b }

type BaseInvalidState interface {
	GetInvalidState() *InvalidState
}

func init() {
	t["BaseInvalidState"] = reflect.TypeOf((*InvalidState)(nil)).Elem()
}

func (b *InvalidVmConfig) GetInvalidVmConfig() *InvalidVmConfig { return b }

type BaseInvalidVmConfig interface {
	GetInvalidVmConfig() *InvalidVmConfig
}

func init() {
	t["BaseInvalidVmConfig"] = reflect.TypeOf((*InvalidVmConfig)(nil)).Elem()
}

func (b *IoFilterInfo) GetIoFilterInfo() *IoFilterInfo { return b }

type BaseIoFilterInfo interface {
	GetIoFilterInfo() *IoFilterInfo
}

func init() {
	t["BaseIoFilterInfo"] = reflect.TypeOf((*IoFilterInfo)(nil)).Elem()
}

func (b *IpAddress) GetIpAddress() *IpAddress { return b }

type BaseIpAddress interface {
	GetIpAddress() *IpAddress
}

func init() {
	t["BaseIpAddress"] = reflect.TypeOf((*IpAddress)(nil)).Elem()
}

func (b *IscsiFault) GetIscsiFault() *IscsiFault { return b }

type BaseIscsiFault interface {
	GetIscsiFault() *IscsiFault
}

func init() {
	t["BaseIscsiFault"] = reflect.TypeOf((*IscsiFault)(nil)).Elem()
}

func (b *LicenseEvent) GetLicenseEvent() *LicenseEvent { return b }

type BaseLicenseEvent interface {
	GetLicenseEvent() *LicenseEvent
}

func init() {
	t["BaseLicenseEvent"] = reflect.TypeOf((*LicenseEvent)(nil)).Elem()
}

func (b *LicenseSource) GetLicenseSource() *LicenseSource { return b }

type BaseLicenseSource interface {
	GetLicenseSource() *LicenseSource
}

func init() {
	t["BaseLicenseSource"] = reflect.TypeOf((*LicenseSource)(nil)).Elem()
}

func (b *MacAddress) GetMacAddress() *MacAddress { return b }

type BaseMacAddress interface {
	GetMacAddress() *MacAddress
}

func init() {
	t["BaseMacAddress"] = reflect.TypeOf((*MacAddress)(nil)).Elem()
}

func (b *MethodFault) GetMethodFault() *MethodFault { return b }

type BaseMethodFault interface {
	GetMethodFault() *MethodFault
}

func init() {
	t["BaseMethodFault"] = reflect.TypeOf((*MethodFault)(nil)).Elem()
}

func (b *MigrationEvent) GetMigrationEvent() *MigrationEvent { return b }

type BaseMigrationEvent interface {
	GetMigrationEvent() *MigrationEvent
}

func init() {
	t["BaseMigrationEvent"] = reflect.TypeOf((*MigrationEvent)(nil)).Elem()
}

func (b *MigrationFault) GetMigrationFault() *MigrationFault { return b }

type BaseMigrationFault interface {
	GetMigrationFault() *MigrationFault
}

func init() {
	t["BaseMigrationFault"] = reflect.TypeOf((*MigrationFault)(nil)).Elem()
}

func (b *MigrationFeatureNotSupported) GetMigrationFeatureNotSupported() *MigrationFeatureNotSupported {
	return b
}

type BaseMigrationFeatureNotSupported interface {
	GetMigrationFeatureNotSupported() *MigrationFeatureNotSupported
}

func init() {
	t["BaseMigrationFeatureNotSupported"] = reflect.TypeOf((*MigrationFeatureNotSupported)(nil)).Elem()
}

func (b *MonthlyTaskScheduler) GetMonthlyTaskScheduler() *MonthlyTaskScheduler { return b }

type BaseMonthlyTaskScheduler interface {
	GetMonthlyTaskScheduler() *MonthlyTaskScheduler
}

func init() {
	t["BaseMonthlyTaskScheduler"] = reflect.TypeOf((*MonthlyTaskScheduler)(nil)).Elem()
}

func (b *NasConfigFault) GetNasConfigFault() *NasConfigFault { return b }

type BaseNasConfigFault interface {
	GetNasConfigFault() *NasConfigFault
}

func init() {
	t["BaseNasConfigFault"] = reflect.TypeOf((*NasConfigFault)(nil)).Elem()
}

func (b *NegatableExpression) GetNegatableExpression() *NegatableExpression { return b }

type BaseNegatableExpression interface {
	GetNegatableExpression() *NegatableExpression
}

func init() {
	t["BaseNegatableExpression"] = reflect.TypeOf((*NegatableExpression)(nil)).Elem()
}

func (b *NetBIOSConfigInfo) GetNetBIOSConfigInfo() *NetBIOSConfigInfo { return b }

type BaseNetBIOSConfigInfo interface {
	GetNetBIOSConfigInfo() *NetBIOSConfigInfo
}

func init() {
	t["BaseNetBIOSConfigInfo"] = reflect.TypeOf((*NetBIOSConfigInfo)(nil)).Elem()
}

func (b *NetworkSummary) GetNetworkSummary() *NetworkSummary { return b }

type BaseNetworkSummary interface {
	GetNetworkSummary() *NetworkSummary
}

func init() {
	t["BaseNetworkSummary"] = reflect.TypeOf((*NetworkSummary)(nil)).Elem()
}

func (b *NoCompatibleHost) GetNoCompatibleHost() *NoCompatibleHost { return b }

type BaseNoCompatibleHost interface {
	GetNoCompatibleHost() *NoCompatibleHost
}

func init() {
	t["BaseNoCompatibleHost"] = reflect.TypeOf((*NoCompatibleHost)(nil)).Elem()
}

func (b *NoPermission) GetNoPermission() *NoPermission { return b }

type BaseNoPermission interface {
	GetNoPermission() *NoPermission
}

func init() {
	t["BaseNoPermission"] = reflect.TypeOf((*NoPermission)(nil)).Elem()
}

func (b *NodeDeploymentSpec) GetNodeDeploymentSpec() *NodeDeploymentSpec { return b }

type BaseNodeDeploymentSpec interface {
	GetNodeDeploymentSpec() *NodeDeploymentSpec
}

func init() {
	t["BaseNodeDeploymentSpec"] = reflect.TypeOf((*NodeDeploymentSpec)(nil)).Elem()
}

func (b *NodeNetworkSpec) GetNodeNetworkSpec() *NodeNetworkSpec { return b }

type BaseNodeNetworkSpec interface {
	GetNodeNetworkSpec() *NodeNetworkSpec
}

func init() {
	t["BaseNodeNetworkSpec"] = reflect.TypeOf((*NodeNetworkSpec)(nil)).Elem()
}

func (b *NotEnoughCpus) GetNotEnoughCpus() *NotEnoughCpus { return b }

type BaseNotEnoughCpus interface {
	GetNotEnoughCpus() *NotEnoughCpus
}

func init() {
	t["BaseNotEnoughCpus"] = reflect.TypeOf((*NotEnoughCpus)(nil)).Elem()
}

func (b *NotEnoughLicenses) GetNotEnoughLicenses() *NotEnoughLicenses { return b }

type BaseNotEnoughLicenses interface {
	GetNotEnoughLicenses() *NotEnoughLicenses
}

func init() {
	t["BaseNotEnoughLicenses"] = reflect.TypeOf((*NotEnoughLicenses)(nil)).Elem()
}

func (b *NotSupported) GetNotSupported() *NotSupported { return b }

type BaseNotSupported interface {
	GetNotSupported() *NotSupported
}

func init() {
	t["BaseNotSupported"] = reflect.TypeOf((*NotSupported)(nil)).Elem()
}

func (b *NotSupportedHost) GetNotSupportedHost() *NotSupportedHost { return b }

type BaseNotSupportedHost interface {
	GetNotSupportedHost() *NotSupportedHost
}

func init() {
	t["BaseNotSupportedHost"] = reflect.TypeOf((*NotSupportedHost)(nil)).Elem()
}

func (b *NotSupportedHostInCluster) GetNotSupportedHostInCluster() *NotSupportedHostInCluster {
	return b
}

type BaseNotSupportedHostInCluster interface {
	GetNotSupportedHostInCluster() *NotSupportedHostInCluster
}

func init() {
	t["BaseNotSupportedHostInCluster"] = reflect.TypeOf((*NotSupportedHostInCluster)(nil)).Elem()
}

func (b *OptionType) GetOptionType() *OptionType { return b }

type BaseOptionType interface {
	GetOptionType() *OptionType
}

func init() {
	t["BaseOptionType"] = reflect.TypeOf((*OptionType)(nil)).Elem()
}

func (b *OptionValue) GetOptionValue() *OptionValue { return b }

type BaseOptionValue interface {
	GetOptionValue() *OptionValue
}

func init() {
	t["BaseOptionValue"] = reflect.TypeOf((*OptionValue)(nil)).Elem()
}

func (b *OvfAttribute) GetOvfAttribute() *OvfAttribute { return b }

type BaseOvfAttribute interface {
	GetOvfAttribute() *OvfAttribute
}

func init() {
	t["BaseOvfAttribute"] = reflect.TypeOf((*OvfAttribute)(nil)).Elem()
}

func (b *OvfConnectedDevice) GetOvfConnectedDevice() *OvfConnectedDevice { return b }

type BaseOvfConnectedDevice interface {
	GetOvfConnectedDevice() *OvfConnectedDevice
}

func init() {
	t["BaseOvfConnectedDevice"] = reflect.TypeOf((*OvfConnectedDevice)(nil)).Elem()
}

func (b *OvfConstraint) GetOvfConstraint() *OvfConstraint { return b }

type BaseOvfConstraint interface {
	GetOvfConstraint() *OvfConstraint
}

func init() {
	t["BaseOvfConstraint"] = reflect.TypeOf((*OvfConstraint)(nil)).Elem()
}

func (b *OvfConsumerCallbackFault) GetOvfConsumerCallbackFault() *OvfConsumerCallbackFault { return b }

type BaseOvfConsumerCallbackFault interface {
	GetOvfConsumerCallbackFault() *OvfConsumerCallbackFault
}

func init() {
	t["BaseOvfConsumerCallbackFault"] = reflect.TypeOf((*OvfConsumerCallbackFault)(nil)).Elem()
}

func (b *OvfElement) GetOvfElement() *OvfElement { return b }

type BaseOvfElement interface {
	GetOvfElement() *OvfElement
}

func init() {
	t["BaseOvfElement"] = reflect.TypeOf((*OvfElement)(nil)).Elem()
}

func (b *OvfExport) GetOvfExport() *OvfExport { return b }

type BaseOvfExport interface {
	GetOvfExport() *OvfExport
}

func init() {
	t["BaseOvfExport"] = reflect.TypeOf((*OvfExport)(nil)).Elem()
}

func (b *OvfFault) GetOvfFault() *OvfFault { return b }

type BaseOvfFault interface {
	GetOvfFault() *OvfFault
}

func init() {
	t["BaseOvfFault"] = reflect.TypeOf((*OvfFault)(nil)).Elem()
}

func (b *OvfHardwareExport) GetOvfHardwareExport() *OvfHardwareExport { return b }

type BaseOvfHardwareExport interface {
	GetOvfHardwareExport() *OvfHardwareExport
}

func init() {
	t["BaseOvfHardwareExport"] = reflect.TypeOf((*OvfHardwareExport)(nil)).Elem()
}

func (b *OvfImport) GetOvfImport() *OvfImport { return b }

type BaseOvfImport interface {
	GetOvfImport() *OvfImport
}

func init() {
	t["BaseOvfImport"] = reflect.TypeOf((*OvfImport)(nil)).Elem()
}

func (b *OvfInvalidPackage) GetOvfInvalidPackage() *OvfInvalidPackage { return b }

type BaseOvfInvalidPackage interface {
	GetOvfInvalidPackage() *OvfInvalidPackage
}

func init() {
	t["BaseOvfInvalidPackage"] = reflect.TypeOf((*OvfInvalidPackage)(nil)).Elem()
}

func (b *OvfInvalidValue) GetOvfInvalidValue() *OvfInvalidValue { return b }

type BaseOvfInvalidValue interface {
	GetOvfInvalidValue() *OvfInvalidValue
}

func init() {
	t["BaseOvfInvalidValue"] = reflect.TypeOf((*OvfInvalidValue)(nil)).Elem()
}

func (b *OvfManagerCommonParams) GetOvfManagerCommonParams() *OvfManagerCommonParams { return b }

type BaseOvfManagerCommonParams interface {
	GetOvfManagerCommonParams() *OvfManagerCommonParams
}

func init() {
	t["BaseOvfManagerCommonParams"] = reflect.TypeOf((*OvfManagerCommonParams)(nil)).Elem()
}

func (b *OvfMissingElement) GetOvfMissingElement() *OvfMissingElement { return b }

type BaseOvfMissingElement interface {
	GetOvfMissingElement() *OvfMissingElement
}

func init() {
	t["BaseOvfMissingElement"] = reflect.TypeOf((*OvfMissingElement)(nil)).Elem()
}

func (b *OvfProperty) GetOvfProperty() *OvfProperty { return b }

type BaseOvfProperty interface {
	GetOvfProperty() *OvfProperty
}

func init() {
	t["BaseOvfProperty"] = reflect.TypeOf((*OvfProperty)(nil)).Elem()
}

func (b *OvfSystemFault) GetOvfSystemFault() *OvfSystemFault { return b }

type BaseOvfSystemFault interface {
	GetOvfSystemFault() *OvfSystemFault
}

func init() {
	t["BaseOvfSystemFault"] = reflect.TypeOf((*OvfSystemFault)(nil)).Elem()
}

func (b *OvfUnsupportedAttribute) GetOvfUnsupportedAttribute() *OvfUnsupportedAttribute { return b }

type BaseOvfUnsupportedAttribute interface {
	GetOvfUnsupportedAttribute() *OvfUnsupportedAttribute
}

func init() {
	t["BaseOvfUnsupportedAttribute"] = reflect.TypeOf((*OvfUnsupportedAttribute)(nil)).Elem()
}

func (b *OvfUnsupportedElement) GetOvfUnsupportedElement() *OvfUnsupportedElement { return b }

type BaseOvfUnsupportedElement interface {
	GetOvfUnsupportedElement() *OvfUnsupportedElement
}

func init() {
	t["BaseOvfUnsupportedElement"] = reflect.TypeOf((*OvfUnsupportedElement)(nil)).Elem()
}

func (b *OvfUnsupportedPackage) GetOvfUnsupportedPackage() *OvfUnsupportedPackage { return b }

type BaseOvfUnsupportedPackage interface {
	GetOvfUnsupportedPackage() *OvfUnsupportedPackage
}

func init() {
	t["BaseOvfUnsupportedPackage"] = reflect.TypeOf((*OvfUnsupportedPackage)(nil)).Elem()
}

func (b *PatchMetadataInvalid) GetPatchMetadataInvalid() *PatchMetadataInvalid { return b }

type BasePatchMetadataInvalid interface {
	GetPatchMetadataInvalid() *PatchMetadataInvalid
}

func init() {
	t["BasePatchMetadataInvalid"] = reflect.TypeOf((*PatchMetadataInvalid)(nil)).Elem()
}

func (b *PatchNotApplicable) GetPatchNotApplicable() *PatchNotApplicable { return b }

type BasePatchNotApplicable interface {
	GetPatchNotApplicable() *PatchNotApplicable
}

func init() {
	t["BasePatchNotApplicable"] = reflect.TypeOf((*PatchNotApplicable)(nil)).Elem()
}

func (b *PerfEntityMetricBase) GetPerfEntityMetricBase() *PerfEntityMetricBase { return b }

type BasePerfEntityMetricBase interface {
	GetPerfEntityMetricBase() *PerfEntityMetricBase
}

func init() {
	t["BasePerfEntityMetricBase"] = reflect.TypeOf((*PerfEntityMetricBase)(nil)).Elem()
}

func (b *PerfMetricSeries) GetPerfMetricSeries() *PerfMetricSeries { return b }

type BasePerfMetricSeries interface {
	GetPerfMetricSeries() *PerfMetricSeries
}

func init() {
	t["BasePerfMetricSeries"] = reflect.TypeOf((*PerfMetricSeries)(nil)).Elem()
}

func (b *PermissionEvent) GetPermissionEvent() *PermissionEvent { return b }

type BasePermissionEvent interface {
	GetPermissionEvent() *PermissionEvent
}

func init() {
	t["BasePermissionEvent"] = reflect.TypeOf((*PermissionEvent)(nil)).Elem()
}

func (b *PhysicalNicHint) GetPhysicalNicHint() *PhysicalNicHint { return b }

type BasePhysicalNicHint interface {
	GetPhysicalNicHint() *PhysicalNicHint
}

func init() {
	t["BasePhysicalNicHint"] = reflect.TypeOf((*PhysicalNicHint)(nil)).Elem()
}

func (b *PlatformConfigFault) GetPlatformConfigFault() *PlatformConfigFault { return b }

type BasePlatformConfigFault interface {
	GetPlatformConfigFault() *PlatformConfigFault
}

func init() {
	t["BasePlatformConfigFault"] = reflect.TypeOf((*PlatformConfigFault)(nil)).Elem()
}

func (b *PolicyOption) GetPolicyOption() *PolicyOption { return b }

type BasePolicyOption interface {
	GetPolicyOption() *PolicyOption
}

func init() {
	t["BasePolicyOption"] = reflect.TypeOf((*PolicyOption)(nil)).Elem()
}

func (b *PortGroupProfile) GetPortGroupProfile() *PortGroupProfile { return b }

type BasePortGroupProfile interface {
	GetPortGroupProfile() *PortGroupProfile
}

func init() {
	t["BasePortGroupProfile"] = reflect.TypeOf((*PortGroupProfile)(nil)).Elem()
}

func (b *ProfileConfigInfo) GetProfileConfigInfo() *ProfileConfigInfo { return b }

type BaseProfileConfigInfo interface {
	GetProfileConfigInfo() *ProfileConfigInfo
}

func init() {
	t["BaseProfileConfigInfo"] = reflect.TypeOf((*ProfileConfigInfo)(nil)).Elem()
}

func (b *ProfileCreateSpec) GetProfileCreateSpec() *ProfileCreateSpec { return b }

type BaseProfileCreateSpec interface {
	GetProfileCreateSpec() *ProfileCreateSpec
}

func init() {
	t["BaseProfileCreateSpec"] = reflect.TypeOf((*ProfileCreateSpec)(nil)).Elem()
}

func (b *ProfileEvent) GetProfileEvent() *ProfileEvent { return b }

type BaseProfileEvent interface {
	GetProfileEvent() *ProfileEvent
}

func init() {
	t["BaseProfileEvent"] = reflect.TypeOf((*ProfileEvent)(nil)).Elem()
}

func (b *ProfileExecuteResult) GetProfileExecuteResult() *ProfileExecuteResult { return b }

type BaseProfileExecuteResult interface {
	GetProfileExecuteResult() *ProfileExecuteResult
}

func init() {
	t["BaseProfileExecuteResult"] = reflect.TypeOf((*ProfileExecuteResult)(nil)).Elem()
}

func (b *ProfileExpression) GetProfileExpression() *ProfileExpression { return b }

type BaseProfileExpression interface {
	GetProfileExpression() *ProfileExpression
}

func init() {
	t["BaseProfileExpression"] = reflect.TypeOf((*ProfileExpression)(nil)).Elem()
}

func (b *ProfilePolicyOptionMetadata) GetProfilePolicyOptionMetadata() *ProfilePolicyOptionMetadata {
	return b
}

type BaseProfilePolicyOptionMetadata interface {
	GetProfilePolicyOptionMetadata() *ProfilePolicyOptionMetadata
}

func init() {
	t["BaseProfilePolicyOptionMetadata"] = reflect.TypeOf((*ProfilePolicyOptionMetadata)(nil)).Elem()
}

func (b *ProfileSerializedCreateSpec) GetProfileSerializedCreateSpec() *ProfileSerializedCreateSpec {
	return b
}

type BaseProfileSerializedCreateSpec interface {
	GetProfileSerializedCreateSpec() *ProfileSerializedCreateSpec
}

func init() {
	t["BaseProfileSerializedCreateSpec"] = reflect.TypeOf((*ProfileSerializedCreateSpec)(nil)).Elem()
}

func (b *RDMNotSupported) GetRDMNotSupported() *RDMNotSupported { return b }

type BaseRDMNotSupported interface {
	GetRDMNotSupported() *RDMNotSupported
}

func init() {
	t["BaseRDMNotSupported"] = reflect.TypeOf((*RDMNotSupported)(nil)).Elem()
}

func (b *RecurrentTaskScheduler) GetRecurrentTaskScheduler() *RecurrentTaskScheduler { return b }

type BaseRecurrentTaskScheduler interface {
	GetRecurrentTaskScheduler() *RecurrentTaskScheduler
}

func init() {
	t["BaseRecurrentTaskScheduler"] = reflect.TypeOf((*RecurrentTaskScheduler)(nil)).Elem()
}

func (b *ReplicationConfigFault) GetReplicationConfigFault() *ReplicationConfigFault { return b }

type BaseReplicationConfigFault interface {
	GetReplicationConfigFault() *ReplicationConfigFault
}

func init() {
	t["BaseReplicationConfigFault"] = reflect.TypeOf((*ReplicationConfigFault)(nil)).Elem()
}

func (b *ReplicationFault) GetReplicationFault() *ReplicationFault { return b }

type BaseReplicationFault interface {
	GetReplicationFault() *ReplicationFault
}

func init() {
	t["BaseReplicationFault"] = reflect.TypeOf((*ReplicationFault)(nil)).Elem()
}

func (b *ReplicationVmFault) GetReplicationVmFault() *ReplicationVmFault { return b }

type BaseReplicationVmFault interface {
	GetReplicationVmFault() *ReplicationVmFault
}

func init() {
	t["BaseReplicationVmFault"] = reflect.TypeOf((*ReplicationVmFault)(nil)).Elem()
}

func (b *ResourceInUse) GetResourceInUse() *ResourceInUse { return b }

type BaseResourceInUse interface {
	GetResourceInUse() *ResourceInUse
}

func init() {
	t["BaseResourceInUse"] = reflect.TypeOf((*ResourceInUse)(nil)).Elem()
}

func (b *ResourcePoolEvent) GetResourcePoolEvent() *ResourcePoolEvent { return b }

type BaseResourcePoolEvent interface {
	GetResourcePoolEvent() *ResourcePoolEvent
}

func init() {
	t["BaseResourcePoolEvent"] = reflect.TypeOf((*ResourcePoolEvent)(nil)).Elem()
}

func (b *ResourcePoolSummary) GetResourcePoolSummary() *ResourcePoolSummary { return b }

type BaseResourcePoolSummary interface {
	GetResourcePoolSummary() *ResourcePoolSummary
}

func init() {
	t["BaseResourcePoolSummary"] = reflect.TypeOf((*ResourcePoolSummary)(nil)).Elem()
}

func (b *RoleEvent) GetRoleEvent() *RoleEvent { return b }

type BaseRoleEvent interface {
	GetRoleEvent() *RoleEvent
}

func init() {
	t["BaseRoleEvent"] = reflect.TypeOf((*RoleEvent)(nil)).Elem()
}

func (b *RuntimeFault) GetRuntimeFault() *RuntimeFault { return b }

type BaseRuntimeFault interface {
	GetRuntimeFault() *RuntimeFault
}

func init() {
	t["BaseRuntimeFault"] = reflect.TypeOf((*RuntimeFault)(nil)).Elem()
}

func (b *ScheduledTaskEvent) GetScheduledTaskEvent() *ScheduledTaskEvent { return b }

type BaseScheduledTaskEvent interface {
	GetScheduledTaskEvent() *ScheduledTaskEvent
}

func init() {
	t["BaseScheduledTaskEvent"] = reflect.TypeOf((*ScheduledTaskEvent)(nil)).Elem()
}

func (b *ScheduledTaskSpec) GetScheduledTaskSpec() *ScheduledTaskSpec { return b }

type BaseScheduledTaskSpec interface {
	GetScheduledTaskSpec() *ScheduledTaskSpec
}

func init() {
	t["BaseScheduledTaskSpec"] = reflect.TypeOf((*ScheduledTaskSpec)(nil)).Elem()
}

func (b *ScsiLun) GetScsiLun() *ScsiLun { return b }

type BaseScsiLun interface {
	GetScsiLun() *ScsiLun
}

func init() {
	t["BaseScsiLun"] = reflect.TypeOf((*ScsiLun)(nil)).Elem()
}

func (b *SecurityError) GetSecurityError() *SecurityError { return b }

type BaseSecurityError interface {
	GetSecurityError() *SecurityError
}

func init() {
	t["BaseSecurityError"] = reflect.TypeOf((*SecurityError)(nil)).Elem()
}

func (b *SelectionSet) GetSelectionSet() *SelectionSet { return b }

type BaseSelectionSet interface {
	GetSelectionSet() *SelectionSet
}

func init() {
	t["BaseSelectionSet"] = reflect.TypeOf((*SelectionSet)(nil)).Elem()
}

func (b *SelectionSpec) GetSelectionSpec() *SelectionSpec { return b }

type BaseSelectionSpec interface {
	GetSelectionSpec() *SelectionSpec
}

func init() {
	t["BaseSelectionSpec"] = reflect.TypeOf((*SelectionSpec)(nil)).Elem()
}

func (b *ServiceLocatorCredential) GetServiceLocatorCredential() *ServiceLocatorCredential { return b }

type BaseServiceLocatorCredential interface {
	GetServiceLocatorCredential() *ServiceLocatorCredential
}

func init() {
	t["BaseServiceLocatorCredential"] = reflect.TypeOf((*ServiceLocatorCredential)(nil)).Elem()
}

func (b *SessionEvent) GetSessionEvent() *SessionEvent { return b }

type BaseSessionEvent interface {
	GetSessionEvent() *SessionEvent
}

func init() {
	t["BaseSessionEvent"] = reflect.TypeOf((*SessionEvent)(nil)).Elem()
}

func (b *SessionManagerServiceRequestSpec) GetSessionManagerServiceRequestSpec() *SessionManagerServiceRequestSpec {
	return b
}

type BaseSessionManagerServiceRequestSpec interface {
	GetSessionManagerServiceRequestSpec() *SessionManagerServiceRequestSpec
}

func init() {
	t["BaseSessionManagerServiceRequestSpec"] = reflect.TypeOf((*SessionManagerServiceRequestSpec)(nil)).Elem()
}

func (b *SnapshotCopyNotSupported) GetSnapshotCopyNotSupported() *SnapshotCopyNotSupported { return b }

type BaseSnapshotCopyNotSupported interface {
	GetSnapshotCopyNotSupported() *SnapshotCopyNotSupported
}

func init() {
	t["BaseSnapshotCopyNotSupported"] = reflect.TypeOf((*SnapshotCopyNotSupported)(nil)).Elem()
}

func (b *SnapshotFault) GetSnapshotFault() *SnapshotFault { return b }

type BaseSnapshotFault interface {
	GetSnapshotFault() *SnapshotFault
}

func init() {
	t["BaseSnapshotFault"] = reflect.TypeOf((*SnapshotFault)(nil)).Elem()
}

func (b *TaskEvent) GetTaskEvent() *TaskEvent { return b }

type BaseTaskEvent interface {
	GetTaskEvent() *TaskEvent
}

func init() {
	t["BaseTaskEvent"] = reflect.TypeOf((*TaskEvent)(nil)).Elem()
}

func (b *TaskInProgress) GetTaskInProgress() *TaskInProgress { return b }

type BaseTaskInProgress interface {
	GetTaskInProgress() *TaskInProgress
}

func init() {
	t["BaseTaskInProgress"] = reflect.TypeOf((*TaskInProgress)(nil)).Elem()
}

func (b *TaskReason) GetTaskReason() *TaskReason { return b }

type BaseTaskReason interface {
	GetTaskReason() *TaskReason
}

func init() {
	t["BaseTaskReason"] = reflect.TypeOf((*TaskReason)(nil)).Elem()
}

func (b *TaskScheduler) GetTaskScheduler() *TaskScheduler { return b }

type BaseTaskScheduler interface {
	GetTaskScheduler() *TaskScheduler
}

func init() {
	t["BaseTaskScheduler"] = reflect.TypeOf((*TaskScheduler)(nil)).Elem()
}

func (b *TemplateUpgradeEvent) GetTemplateUpgradeEvent() *TemplateUpgradeEvent { return b }

type BaseTemplateUpgradeEvent interface {
	GetTemplateUpgradeEvent() *TemplateUpgradeEvent
}

func init() {
	t["BaseTemplateUpgradeEvent"] = reflect.TypeOf((*TemplateUpgradeEvent)(nil)).Elem()
}

func (b *Timedout) GetTimedout() *Timedout { return b }

type BaseTimedout interface {
	GetTimedout() *Timedout
}

func init() {
	t["BaseTimedout"] = reflect.TypeOf((*Timedout)(nil)).Elem()
}

func (b *TypeDescription) GetTypeDescription() *TypeDescription { return b }

type BaseTypeDescription interface {
	GetTypeDescription() *TypeDescription
}

func init() {
	t["BaseTypeDescription"] = reflect.TypeOf((*TypeDescription)(nil)).Elem()
}

func (b *UnsupportedDatastore) GetUnsupportedDatastore() *UnsupportedDatastore { return b }

type BaseUnsupportedDatastore interface {
	GetUnsupportedDatastore() *UnsupportedDatastore
}

func init() {
	t["BaseUnsupportedDatastore"] = reflect.TypeOf((*UnsupportedDatastore)(nil)).Elem()
}

func (b *UpgradeEvent) GetUpgradeEvent() *UpgradeEvent { return b }

type BaseUpgradeEvent interface {
	GetUpgradeEvent() *UpgradeEvent
}

func init() {
	t["BaseUpgradeEvent"] = reflect.TypeOf((*UpgradeEvent)(nil)).Elem()
}

func (b *UserSearchResult) GetUserSearchResult() *UserSearchResult { return b }

type BaseUserSearchResult interface {
	GetUserSearchResult() *UserSearchResult
}

func init() {
	t["BaseUserSearchResult"] = reflect.TypeOf((*UserSearchResult)(nil)).Elem()
}

func (b *VAppConfigFault) GetVAppConfigFault() *VAppConfigFault { return b }

type BaseVAppConfigFault interface {
	GetVAppConfigFault() *VAppConfigFault
}

func init() {
	t["BaseVAppConfigFault"] = reflect.TypeOf((*VAppConfigFault)(nil)).Elem()
}

func (b *VAppPropertyFault) GetVAppPropertyFault() *VAppPropertyFault { return b }

type BaseVAppPropertyFault interface {
	GetVAppPropertyFault() *VAppPropertyFault
}

func init() {
	t["BaseVAppPropertyFault"] = reflect.TypeOf((*VAppPropertyFault)(nil)).Elem()
}

func (b *VMotionInterfaceIssue) GetVMotionInterfaceIssue() *VMotionInterfaceIssue { return b }

type BaseVMotionInterfaceIssue interface {
	GetVMotionInterfaceIssue() *VMotionInterfaceIssue
}

func init() {
	t["BaseVMotionInterfaceIssue"] = reflect.TypeOf((*VMotionInterfaceIssue)(nil)).Elem()
}

func (b *VMwareDVSHealthCheckConfig) GetVMwareDVSHealthCheckConfig() *VMwareDVSHealthCheckConfig {
	return b
}

type BaseVMwareDVSHealthCheckConfig interface {
	GetVMwareDVSHealthCheckConfig() *VMwareDVSHealthCheckConfig
}

func init() {
	t["BaseVMwareDVSHealthCheckConfig"] = reflect.TypeOf((*VMwareDVSHealthCheckConfig)(nil)).Elem()
}

func (b *VimFault) GetVimFault() *VimFault { return b }

type BaseVimFault interface {
	GetVimFault() *VimFault
}

func init() {
	t["BaseVimFault"] = reflect.TypeOf((*VimFault)(nil)).Elem()
}

func (b *VirtualController) GetVirtualController() *VirtualController { return b }

type BaseVirtualController interface {
	GetVirtualController() *VirtualController
}

func init() {
	t["BaseVirtualController"] = reflect.TypeOf((*VirtualController)(nil)).Elem()
}

func (b *VirtualControllerOption) GetVirtualControllerOption() *VirtualControllerOption { return b }

type BaseVirtualControllerOption interface {
	GetVirtualControllerOption() *VirtualControllerOption
}

func init() {
	t["BaseVirtualControllerOption"] = reflect.TypeOf((*VirtualControllerOption)(nil)).Elem()
}

func (b *VirtualDevice) GetVirtualDevice() *VirtualDevice { return b }

type BaseVirtualDevice interface {
	GetVirtualDevice() *VirtualDevice
}

func init() {
	t["BaseVirtualDevice"] = reflect.TypeOf((*VirtualDevice)(nil)).Elem()
}

func (b *VirtualDeviceBackingInfo) GetVirtualDeviceBackingInfo() *VirtualDeviceBackingInfo { return b }

type BaseVirtualDeviceBackingInfo interface {
	GetVirtualDeviceBackingInfo() *VirtualDeviceBackingInfo
}

func init() {
	t["BaseVirtualDeviceBackingInfo"] = reflect.TypeOf((*VirtualDeviceBackingInfo)(nil)).Elem()
}

func (b *VirtualDeviceBackingOption) GetVirtualDeviceBackingOption() *VirtualDeviceBackingOption {
	return b
}

type BaseVirtualDeviceBackingOption interface {
	GetVirtualDeviceBackingOption() *VirtualDeviceBackingOption
}

func init() {
	t["BaseVirtualDeviceBackingOption"] = reflect.TypeOf((*VirtualDeviceBackingOption)(nil)).Elem()
}

func (b *VirtualDeviceBusSlotInfo) GetVirtualDeviceBusSlotInfo() *VirtualDeviceBusSlotInfo { return b }

type BaseVirtualDeviceBusSlotInfo interface {
	GetVirtualDeviceBusSlotInfo() *VirtualDeviceBusSlotInfo
}

func init() {
	t["BaseVirtualDeviceBusSlotInfo"] = reflect.TypeOf((*VirtualDeviceBusSlotInfo)(nil)).Elem()
}

func (b *VirtualDeviceConfigSpec) GetVirtualDeviceConfigSpec() *VirtualDeviceConfigSpec { return b }

type BaseVirtualDeviceConfigSpec interface {
	GetVirtualDeviceConfigSpec() *VirtualDeviceConfigSpec
}

func init() {
	t["BaseVirtualDeviceConfigSpec"] = reflect.TypeOf((*VirtualDeviceConfigSpec)(nil)).Elem()
}

func (b *VirtualDeviceDeviceBackingInfo) GetVirtualDeviceDeviceBackingInfo() *VirtualDeviceDeviceBackingInfo {
	return b
}

type BaseVirtualDeviceDeviceBackingInfo interface {
	GetVirtualDeviceDeviceBackingInfo() *VirtualDeviceDeviceBackingInfo
}

func init() {
	t["BaseVirtualDeviceDeviceBackingInfo"] = reflect.TypeOf((*VirtualDeviceDeviceBackingInfo)(nil)).Elem()
}

func (b *VirtualDeviceDeviceBackingOption) GetVirtualDeviceDeviceBackingOption() *VirtualDeviceDeviceBackingOption {
	return b
}

type BaseVirtualDeviceDeviceBackingOption interface {
	GetVirtualDeviceDeviceBackingOption() *VirtualDeviceDeviceBackingOption
}

func init() {
	t["BaseVirtualDeviceDeviceBackingOption"] = reflect.TypeOf((*VirtualDeviceDeviceBackingOption)(nil)).Elem()
}

func (b *VirtualDeviceFileBackingInfo) GetVirtualDeviceFileBackingInfo() *VirtualDeviceFileBackingInfo {
	return b
}

type BaseVirtualDeviceFileBackingInfo interface {
	GetVirtualDeviceFileBackingInfo() *VirtualDeviceFileBackingInfo
}

func init() {
	t["BaseVirtualDeviceFileBackingInfo"] = reflect.TypeOf((*VirtualDeviceFileBackingInfo)(nil)).Elem()
}

func (b *VirtualDeviceFileBackingOption) GetVirtualDeviceFileBackingOption() *VirtualDeviceFileBackingOption {
	return b
}

type BaseVirtualDeviceFileBackingOption interface {
	GetVirtualDeviceFileBackingOption() *VirtualDeviceFileBackingOption
}

func init() {
	t["BaseVirtualDeviceFileBackingOption"] = reflect.TypeOf((*VirtualDeviceFileBackingOption)(nil)).Elem()
}

func (b *VirtualDeviceOption) GetVirtualDeviceOption() *VirtualDeviceOption { return b }

type BaseVirtualDeviceOption interface {
	GetVirtualDeviceOption() *VirtualDeviceOption
}

func init() {
	t["BaseVirtualDeviceOption"] = reflect.TypeOf((*VirtualDeviceOption)(nil)).Elem()
}

func (b *VirtualDevicePciBusSlotInfo) GetVirtualDevicePciBusSlotInfo() *VirtualDevicePciBusSlotInfo {
	return b
}

type BaseVirtualDevicePciBusSlotInfo interface {
	GetVirtualDevicePciBusSlotInfo() *VirtualDevicePciBusSlotInfo
}

func init() {
	t["BaseVirtualDevicePciBusSlotInfo"] = reflect.TypeOf((*VirtualDevicePciBusSlotInfo)(nil)).Elem()
}

func (b *VirtualDevicePipeBackingInfo) GetVirtualDevicePipeBackingInfo() *VirtualDevicePipeBackingInfo {
	return b
}

type BaseVirtualDevicePipeBackingInfo interface {
	GetVirtualDevicePipeBackingInfo() *VirtualDevicePipeBackingInfo
}

func init() {
	t["BaseVirtualDevicePipeBackingInfo"] = reflect.TypeOf((*VirtualDevicePipeBackingInfo)(nil)).Elem()
}

func (b *VirtualDevicePipeBackingOption) GetVirtualDevicePipeBackingOption() *VirtualDevicePipeBackingOption {
	return b
}

type BaseVirtualDevicePipeBackingOption interface {
	GetVirtualDevicePipeBackingOption() *VirtualDevicePipeBackingOption
}

func init() {
	t["BaseVirtualDevicePipeBackingOption"] = reflect.TypeOf((*VirtualDevicePipeBackingOption)(nil)).Elem()
}

func (b *VirtualDeviceRemoteDeviceBackingInfo) GetVirtualDeviceRemoteDeviceBackingInfo() *VirtualDeviceRemoteDeviceBackingInfo {
	return b
}

type BaseVirtualDeviceRemoteDeviceBackingInfo interface {
	GetVirtualDeviceRemoteDeviceBackingInfo() *VirtualDeviceRemoteDeviceBackingInfo
}

func init() {
	t["BaseVirtualDeviceRemoteDeviceBackingInfo"] = reflect.TypeOf((*VirtualDeviceRemoteDeviceBackingInfo)(nil)).Elem()
}

func (b *VirtualDeviceRemoteDeviceBackingOption) GetVirtualDeviceRemoteDeviceBackingOption() *VirtualDeviceRemoteDeviceBackingOption {
	return b
}

type BaseVirtualDeviceRemoteDeviceBackingOption interface {
	GetVirtualDeviceRemoteDeviceBackingOption() *VirtualDeviceRemoteDeviceBackingOption
}

func init() {
	t["BaseVirtualDeviceRemoteDeviceBackingOption"] = reflect.TypeOf((*VirtualDeviceRemoteDeviceBackingOption)(nil)).Elem()
}

func (b *VirtualDeviceURIBackingInfo) GetVirtualDeviceURIBackingInfo() *VirtualDeviceURIBackingInfo {
	return b
}

type BaseVirtualDeviceURIBackingInfo interface {
	GetVirtualDeviceURIBackingInfo() *VirtualDeviceURIBackingInfo
}

func init() {
	t["BaseVirtualDeviceURIBackingInfo"] = reflect.TypeOf((*VirtualDeviceURIBackingInfo)(nil)).Elem()
}

func (b *VirtualDeviceURIBackingOption) GetVirtualDeviceURIBackingOption() *VirtualDeviceURIBackingOption {
	return b
}

type BaseVirtualDeviceURIBackingOption interface {
	GetVirtualDeviceURIBackingOption() *VirtualDeviceURIBackingOption
}

func init() {
	t["BaseVirtualDeviceURIBackingOption"] = reflect.TypeOf((*VirtualDeviceURIBackingOption)(nil)).Elem()
}

func (b *VirtualDiskRawDiskVer2BackingInfo) GetVirtualDiskRawDiskVer2BackingInfo() *VirtualDiskRawDiskVer2BackingInfo {
	return b
}

type BaseVirtualDiskRawDiskVer2BackingInfo interface {
	GetVirtualDiskRawDiskVer2BackingInfo() *VirtualDiskRawDiskVer2BackingInfo
}

func init() {
	t["BaseVirtualDiskRawDiskVer2BackingInfo"] = reflect.TypeOf((*VirtualDiskRawDiskVer2BackingInfo)(nil)).Elem()
}

func (b *VirtualDiskRawDiskVer2BackingOption) GetVirtualDiskRawDiskVer2BackingOption() *VirtualDiskRawDiskVer2BackingOption {
	return b
}

type BaseVirtualDiskRawDiskVer2BackingOption interface {
	GetVirtualDiskRawDiskVer2BackingOption() *VirtualDiskRawDiskVer2BackingOption
}

func init() {
	t["BaseVirtualDiskRawDiskVer2BackingOption"] = reflect.TypeOf((*VirtualDiskRawDiskVer2BackingOption)(nil)).Elem()
}

func (b *VirtualDiskSpec) GetVirtualDiskSpec() *VirtualDiskSpec { return b }

type BaseVirtualDiskSpec interface {
	GetVirtualDiskSpec() *VirtualDiskSpec
}

func init() {
	t["BaseVirtualDiskSpec"] = reflect.TypeOf((*VirtualDiskSpec)(nil)).Elem()
}

func (b *VirtualEthernetCard) GetVirtualEthernetCard() *VirtualEthernetCard { return b }

type BaseVirtualEthernetCard interface {
	GetVirtualEthernetCard() *VirtualEthernetCard
}

func init() {
	t["BaseVirtualEthernetCard"] = reflect.TypeOf((*VirtualEthernetCard)(nil)).Elem()
}

func (b *VirtualEthernetCardOption) GetVirtualEthernetCardOption() *VirtualEthernetCardOption {
	return b
}

type BaseVirtualEthernetCardOption interface {
	GetVirtualEthernetCardOption() *VirtualEthernetCardOption
}

func init() {
	t["BaseVirtualEthernetCardOption"] = reflect.TypeOf((*VirtualEthernetCardOption)(nil)).Elem()
}

func (b *VirtualHardwareCompatibilityIssue) GetVirtualHardwareCompatibilityIssue() *VirtualHardwareCompatibilityIssue {
	return b
}

type BaseVirtualHardwareCompatibilityIssue interface {
	GetVirtualHardwareCompatibilityIssue() *VirtualHardwareCompatibilityIssue
}

func init() {
	t["BaseVirtualHardwareCompatibilityIssue"] = reflect.TypeOf((*VirtualHardwareCompatibilityIssue)(nil)).Elem()
}

func (b *VirtualMachineBootOptionsBootableDevice) GetVirtualMachineBootOptionsBootableDevice() *VirtualMachineBootOptionsBootableDevice {
	return b
}

type BaseVirtualMachineBootOptionsBootableDevice interface {
	GetVirtualMachineBootOptionsBootableDevice() *VirtualMachineBootOptionsBootableDevice
}

func init() {
	t["BaseVirtualMachineBootOptionsBootableDevice"] = reflect.TypeOf((*VirtualMachineBootOptionsBootableDevice)(nil)).Elem()
}

func (b *VirtualMachineDeviceRuntimeInfoDeviceRuntimeState) GetVirtualMachineDeviceRuntimeInfoDeviceRuntimeState() *VirtualMachineDeviceRuntimeInfoDeviceRuntimeState {
	return b
}

type BaseVirtualMachineDeviceRuntimeInfoDeviceRuntimeState interface {
	GetVirtualMachineDeviceRuntimeInfoDeviceRuntimeState() *VirtualMachineDeviceRuntimeInfoDeviceRuntimeState
}

func init() {
	t["BaseVirtualMachineDeviceRuntimeInfoDeviceRuntimeState"] = reflect.TypeOf((*VirtualMachineDeviceRuntimeInfoDeviceRuntimeState)(nil)).Elem()
}

func (b *VirtualMachineDiskDeviceInfo) GetVirtualMachineDiskDeviceInfo() *VirtualMachineDiskDeviceInfo {
	return b
}

type BaseVirtualMachineDiskDeviceInfo interface {
	GetVirtualMachineDiskDeviceInfo() *VirtualMachineDiskDeviceInfo
}

func init() {
	t["BaseVirtualMachineDiskDeviceInfo"] = reflect.TypeOf((*VirtualMachineDiskDeviceInfo)(nil)).Elem()
}

func (b *VirtualMachineGuestQuiesceSpec) GetVirtualMachineGuestQuiesceSpec() *VirtualMachineGuestQuiesceSpec {
	return b
}

type BaseVirtualMachineGuestQuiesceSpec interface {
	GetVirtualMachineGuestQuiesceSpec() *VirtualMachineGuestQuiesceSpec
}

func init() {
	t["BaseVirtualMachineGuestQuiesceSpec"] = reflect.TypeOf((*VirtualMachineGuestQuiesceSpec)(nil)).Elem()
}

func (b *VirtualMachinePciPassthroughInfo) GetVirtualMachinePciPassthroughInfo() *VirtualMachinePciPassthroughInfo {
	return b
}

type BaseVirtualMachinePciPassthroughInfo interface {
	GetVirtualMachinePciPassthroughInfo() *VirtualMachinePciPassthroughInfo
}

func init() {
	t["BaseVirtualMachinePciPassthroughInfo"] = reflect.TypeOf((*VirtualMachinePciPassthroughInfo)(nil)).Elem()
}

func (b *VirtualMachineProfileSpec) GetVirtualMachineProfileSpec() *VirtualMachineProfileSpec {
	return b
}

type BaseVirtualMachineProfileSpec interface {
	GetVirtualMachineProfileSpec() *VirtualMachineProfileSpec
}

func init() {
	t["BaseVirtualMachineProfileSpec"] = reflect.TypeOf((*VirtualMachineProfileSpec)(nil)).Elem()
}

func (b *VirtualMachineSriovDevicePoolInfo) GetVirtualMachineSriovDevicePoolInfo() *VirtualMachineSriovDevicePoolInfo {
	return b
}

type BaseVirtualMachineSriovDevicePoolInfo interface {
	GetVirtualMachineSriovDevicePoolInfo() *VirtualMachineSriovDevicePoolInfo
}

func init() {
	t["BaseVirtualMachineSriovDevicePoolInfo"] = reflect.TypeOf((*VirtualMachineSriovDevicePoolInfo)(nil)).Elem()
}

func (b *VirtualMachineTargetInfo) GetVirtualMachineTargetInfo() *VirtualMachineTargetInfo { return b }

type BaseVirtualMachineTargetInfo interface {
	GetVirtualMachineTargetInfo() *VirtualMachineTargetInfo
}

func init() {
	t["BaseVirtualMachineTargetInfo"] = reflect.TypeOf((*VirtualMachineTargetInfo)(nil)).Elem()
}

func (b *VirtualPCIPassthroughPluginBackingInfo) GetVirtualPCIPassthroughPluginBackingInfo() *VirtualPCIPassthroughPluginBackingInfo {
	return b
}

type BaseVirtualPCIPassthroughPluginBackingInfo interface {
	GetVirtualPCIPassthroughPluginBackingInfo() *VirtualPCIPassthroughPluginBackingInfo
}

func init() {
	t["BaseVirtualPCIPassthroughPluginBackingInfo"] = reflect.TypeOf((*VirtualPCIPassthroughPluginBackingInfo)(nil)).Elem()
}

func (b *VirtualPCIPassthroughPluginBackingOption) GetVirtualPCIPassthroughPluginBackingOption() *VirtualPCIPassthroughPluginBackingOption {
	return b
}

type BaseVirtualPCIPassthroughPluginBackingOption interface {
	GetVirtualPCIPassthroughPluginBackingOption() *VirtualPCIPassthroughPluginBackingOption
}

func init() {
	t["BaseVirtualPCIPassthroughPluginBackingOption"] = reflect.TypeOf((*VirtualPCIPassthroughPluginBackingOption)(nil)).Elem()
}

func (b *VirtualSATAController) GetVirtualSATAController() *VirtualSATAController { return b }

type BaseVirtualSATAController interface {
	GetVirtualSATAController() *VirtualSATAController
}

func init() {
	t["BaseVirtualSATAController"] = reflect.TypeOf((*VirtualSATAController)(nil)).Elem()
}

func (b *VirtualSATAControllerOption) GetVirtualSATAControllerOption() *VirtualSATAControllerOption {
	return b
}

type BaseVirtualSATAControllerOption interface {
	GetVirtualSATAControllerOption() *VirtualSATAControllerOption
}

func init() {
	t["BaseVirtualSATAControllerOption"] = reflect.TypeOf((*VirtualSATAControllerOption)(nil)).Elem()
}

func (b *VirtualSCSIController) GetVirtualSCSIController() *VirtualSCSIController { return b }

type BaseVirtualSCSIController interface {
	GetVirtualSCSIController() *VirtualSCSIController
}

func init() {
	t["BaseVirtualSCSIController"] = reflect.TypeOf((*VirtualSCSIController)(nil)).Elem()
}

func (b *VirtualSCSIControllerOption) GetVirtualSCSIControllerOption() *VirtualSCSIControllerOption {
	return b
}

type BaseVirtualSCSIControllerOption interface {
	GetVirtualSCSIControllerOption() *VirtualSCSIControllerOption
}

func init() {
	t["BaseVirtualSCSIControllerOption"] = reflect.TypeOf((*VirtualSCSIControllerOption)(nil)).Elem()
}

func (b *VirtualSoundCard) GetVirtualSoundCard() *VirtualSoundCard { return b }

type BaseVirtualSoundCard interface {
	GetVirtualSoundCard() *VirtualSoundCard
}

func init() {
	t["BaseVirtualSoundCard"] = reflect.TypeOf((*VirtualSoundCard)(nil)).Elem()
}

func (b *VirtualSoundCardOption) GetVirtualSoundCardOption() *VirtualSoundCardOption { return b }

type BaseVirtualSoundCardOption interface {
	GetVirtualSoundCardOption() *VirtualSoundCardOption
}

func init() {
	t["BaseVirtualSoundCardOption"] = reflect.TypeOf((*VirtualSoundCardOption)(nil)).Elem()
}

func (b *VirtualVmxnet) GetVirtualVmxnet() *VirtualVmxnet { return b }

type BaseVirtualVmxnet interface {
	GetVirtualVmxnet() *VirtualVmxnet
}

func init() {
	t["BaseVirtualVmxnet"] = reflect.TypeOf((*VirtualVmxnet)(nil)).Elem()
}

func (b *VirtualVmxnet3) GetVirtualVmxnet3() *VirtualVmxnet3 { return b }

type BaseVirtualVmxnet3 interface {
	GetVirtualVmxnet3() *VirtualVmxnet3
}

func init() {
	t["BaseVirtualVmxnet3"] = reflect.TypeOf((*VirtualVmxnet3)(nil)).Elem()
}

func (b *VirtualVmxnet3Option) GetVirtualVmxnet3Option() *VirtualVmxnet3Option { return b }

type BaseVirtualVmxnet3Option interface {
	GetVirtualVmxnet3Option() *VirtualVmxnet3Option
}

func init() {
	t["BaseVirtualVmxnet3Option"] = reflect.TypeOf((*VirtualVmxnet3Option)(nil)).Elem()
}

func (b *VirtualVmxnetOption) GetVirtualVmxnetOption() *VirtualVmxnetOption { return b }

type BaseVirtualVmxnetOption interface {
	GetVirtualVmxnetOption() *VirtualVmxnetOption
}

func init() {
	t["BaseVirtualVmxnetOption"] = reflect.TypeOf((*VirtualVmxnetOption)(nil)).Elem()
}

func (b *VmCloneEvent) GetVmCloneEvent() *VmCloneEvent { return b }

type BaseVmCloneEvent interface {
	GetVmCloneEvent() *VmCloneEvent
}

func init() {
	t["BaseVmCloneEvent"] = reflect.TypeOf((*VmCloneEvent)(nil)).Elem()
}

func (b *VmConfigFault) GetVmConfigFault() *VmConfigFault { return b }

type BaseVmConfigFault interface {
	GetVmConfigFault() *VmConfigFault
}

func init() {
	t["BaseVmConfigFault"] = reflect.TypeOf((*VmConfigFault)(nil)).Elem()
}

func (b *VmConfigFileInfo) GetVmConfigFileInfo() *VmConfigFileInfo { return b }

type BaseVmConfigFileInfo interface {
	GetVmConfigFileInfo() *VmConfigFileInfo
}

func init() {
	t["BaseVmConfigFileInfo"] = reflect.TypeOf((*VmConfigFileInfo)(nil)).Elem()
}

func (b *VmConfigFileQuery) GetVmConfigFileQuery() *VmConfigFileQuery { return b }

type BaseVmConfigFileQuery interface {
	GetVmConfigFileQuery() *VmConfigFileQuery
}

func init() {
	t["BaseVmConfigFileQuery"] = reflect.TypeOf((*VmConfigFileQuery)(nil)).Elem()
}

func (b *VmConfigInfo) GetVmConfigInfo() *VmConfigInfo { return b }

type BaseVmConfigInfo interface {
	GetVmConfigInfo() *VmConfigInfo
}

func init() {
	t["BaseVmConfigInfo"] = reflect.TypeOf((*VmConfigInfo)(nil)).Elem()
}

func (b *VmConfigSpec) GetVmConfigSpec() *VmConfigSpec { return b }

type BaseVmConfigSpec interface {
	GetVmConfigSpec() *VmConfigSpec
}

func init() {
	t["BaseVmConfigSpec"] = reflect.TypeOf((*VmConfigSpec)(nil)).Elem()
}

func (b *VmDasBeingResetEvent) GetVmDasBeingResetEvent() *VmDasBeingResetEvent { return b }

type BaseVmDasBeingResetEvent interface {
	GetVmDasBeingResetEvent() *VmDasBeingResetEvent
}

func init() {
	t["BaseVmDasBeingResetEvent"] = reflect.TypeOf((*VmDasBeingResetEvent)(nil)).Elem()
}

func (b *VmEvent) GetVmEvent() *VmEvent { return b }

type BaseVmEvent interface {
	GetVmEvent() *VmEvent
}

func init() {
	t["BaseVmEvent"] = reflect.TypeOf((*VmEvent)(nil)).Elem()
}

func (b *VmFaultToleranceIssue) GetVmFaultToleranceIssue() *VmFaultToleranceIssue { return b }

type BaseVmFaultToleranceIssue interface {
	GetVmFaultToleranceIssue() *VmFaultToleranceIssue
}

func init() {
	t["BaseVmFaultToleranceIssue"] = reflect.TypeOf((*VmFaultToleranceIssue)(nil)).Elem()
}

func (b *VmMigratedEvent) GetVmMigratedEvent() *VmMigratedEvent { return b }

type BaseVmMigratedEvent interface {
	GetVmMigratedEvent() *VmMigratedEvent
}

func init() {
	t["BaseVmMigratedEvent"] = reflect.TypeOf((*VmMigratedEvent)(nil)).Elem()
}

func (b *VmPoweredOffEvent) GetVmPoweredOffEvent() *VmPoweredOffEvent { return b }

type BaseVmPoweredOffEvent interface {
	GetVmPoweredOffEvent() *VmPoweredOffEvent
}

func init() {
	t["BaseVmPoweredOffEvent"] = reflect.TypeOf((*VmPoweredOffEvent)(nil)).Elem()
}

func (b *VmPoweredOnEvent) GetVmPoweredOnEvent() *VmPoweredOnEvent { return b }

type BaseVmPoweredOnEvent interface {
	GetVmPoweredOnEvent() *VmPoweredOnEvent
}

func init() {
	t["BaseVmPoweredOnEvent"] = reflect.TypeOf((*VmPoweredOnEvent)(nil)).Elem()
}

func (b *VmRelocateSpecEvent) GetVmRelocateSpecEvent() *VmRelocateSpecEvent { return b }

type BaseVmRelocateSpecEvent interface {
	GetVmRelocateSpecEvent() *VmRelocateSpecEvent
}

func init() {
	t["BaseVmRelocateSpecEvent"] = reflect.TypeOf((*VmRelocateSpecEvent)(nil)).Elem()
}

func (b *VmStartingEvent) GetVmStartingEvent() *VmStartingEvent { return b }

type BaseVmStartingEvent interface {
	GetVmStartingEvent() *VmStartingEvent
}

func init() {
	t["BaseVmStartingEvent"] = reflect.TypeOf((*VmStartingEvent)(nil)).Elem()
}

func (b *VmToolsUpgradeFault) GetVmToolsUpgradeFault() *VmToolsUpgradeFault { return b }

type BaseVmToolsUpgradeFault interface {
	GetVmToolsUpgradeFault() *VmToolsUpgradeFault
}

func init() {
	t["BaseVmToolsUpgradeFault"] = reflect.TypeOf((*VmToolsUpgradeFault)(nil)).Elem()
}

func (b *VmfsDatastoreBaseOption) GetVmfsDatastoreBaseOption() *VmfsDatastoreBaseOption { return b }

type BaseVmfsDatastoreBaseOption interface {
	GetVmfsDatastoreBaseOption() *VmfsDatastoreBaseOption
}

func init() {
	t["BaseVmfsDatastoreBaseOption"] = reflect.TypeOf((*VmfsDatastoreBaseOption)(nil)).Elem()
}

func (b *VmfsDatastoreSingleExtentOption) GetVmfsDatastoreSingleExtentOption() *VmfsDatastoreSingleExtentOption {
	return b
}

type BaseVmfsDatastoreSingleExtentOption interface {
	GetVmfsDatastoreSingleExtentOption() *VmfsDatastoreSingleExtentOption
}

func init() {
	t["BaseVmfsDatastoreSingleExtentOption"] = reflect.TypeOf((*VmfsDatastoreSingleExtentOption)(nil)).Elem()
}

func (b *VmfsDatastoreSpec) GetVmfsDatastoreSpec() *VmfsDatastoreSpec { return b }

type BaseVmfsDatastoreSpec interface {
	GetVmfsDatastoreSpec() *VmfsDatastoreSpec
}

func init() {
	t["BaseVmfsDatastoreSpec"] = reflect.TypeOf((*VmfsDatastoreSpec)(nil)).Elem()
}

func (b *VmfsMountFault) GetVmfsMountFault() *VmfsMountFault { return b }

type BaseVmfsMountFault interface {
	GetVmfsMountFault() *VmfsMountFault
}

func init() {
	t["BaseVmfsMountFault"] = reflect.TypeOf((*VmfsMountFault)(nil)).Elem()
}

func (b *VmwareDistributedVirtualSwitchVlanSpec) GetVmwareDistributedVirtualSwitchVlanSpec() *VmwareDistributedVirtualSwitchVlanSpec {
	return b
}

type BaseVmwareDistributedVirtualSwitchVlanSpec interface {
	GetVmwareDistributedVirtualSwitchVlanSpec() *VmwareDistributedVirtualSwitchVlanSpec
}

func init() {
	t["BaseVmwareDistributedVirtualSwitchVlanSpec"] = reflect.TypeOf((*VmwareDistributedVirtualSwitchVlanSpec)(nil)).Elem()
}

func (b *VsanDiskFault) GetVsanDiskFault() *VsanDiskFault { return b }

type BaseVsanDiskFault interface {
	GetVsanDiskFault() *VsanDiskFault
}

func init() {
	t["BaseVsanDiskFault"] = reflect.TypeOf((*VsanDiskFault)(nil)).Elem()
}

func (b *VsanFault) GetVsanFault() *VsanFault { return b }

type BaseVsanFault interface {
	GetVsanFault() *VsanFault
}

func init() {
	t["BaseVsanFault"] = reflect.TypeOf((*VsanFault)(nil)).Elem()
}

func (b *VsanUpgradeSystemPreflightCheckIssue) GetVsanUpgradeSystemPreflightCheckIssue() *VsanUpgradeSystemPreflightCheckIssue {
	return b
}

type BaseVsanUpgradeSystemPreflightCheckIssue interface {
	GetVsanUpgradeSystemPreflightCheckIssue() *VsanUpgradeSystemPreflightCheckIssue
}

func init() {
	t["BaseVsanUpgradeSystemPreflightCheckIssue"] = reflect.TypeOf((*VsanUpgradeSystemPreflightCheckIssue)(nil)).Elem()
}

func (b *VsanUpgradeSystemUpgradeHistoryItem) GetVsanUpgradeSystemUpgradeHistoryItem() *VsanUpgradeSystemUpgradeHistoryItem {
	return b
}

type BaseVsanUpgradeSystemUpgradeHistoryItem interface {
	GetVsanUpgradeSystemUpgradeHistoryItem() *VsanUpgradeSystemUpgradeHistoryItem
}

func init() {
	t["BaseVsanUpgradeSystemUpgradeHistoryItem"] = reflect.TypeOf((*VsanUpgradeSystemUpgradeHistoryItem)(nil)).Elem()
}

func (b *VslmCreateSpecBackingSpec) GetVslmCreateSpecBackingSpec() *VslmCreateSpecBackingSpec {
	return b
}

type BaseVslmCreateSpecBackingSpec interface {
	GetVslmCreateSpecBackingSpec() *VslmCreateSpecBackingSpec
}

func init() {
	t["BaseVslmCreateSpecBackingSpec"] = reflect.TypeOf((*VslmCreateSpecBackingSpec)(nil)).Elem()
}

func (b *VslmMigrateSpec) GetVslmMigrateSpec() *VslmMigrateSpec { return b }

type BaseVslmMigrateSpec interface {
	GetVslmMigrateSpec() *VslmMigrateSpec
}

func init() {
	t["BaseVslmMigrateSpec"] = reflect.TypeOf((*VslmMigrateSpec)(nil)).Elem()
}
