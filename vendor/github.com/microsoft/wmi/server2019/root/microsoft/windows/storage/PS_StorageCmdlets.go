// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// PS_StorageCmdlets struct
type PS_StorageCmdlets struct {
	*cim.WmiInstance
}

func NewPS_StorageCmdletsEx1(instance *cim.WmiInstance) (newInstance *PS_StorageCmdlets, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &PS_StorageCmdlets{
		WmiInstance: tmp,
	}
	return
}

func NewPS_StorageCmdletsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *PS_StorageCmdlets, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &PS_StorageCmdlets{
		WmiInstance: tmp,
	}
	return
}

//

// <param name="Guid" type="string "></param>
// <param name="InputObject" type="MSFT_Disk []"></param>
// <param name="IsOffline" type="bool "></param>
// <param name="IsReadOnly" type="bool "></param>
// <param name="Number" type="uint32 "></param>
// <param name="PartitionStyle" type="uint16 "></param>
// <param name="Path" type="string "></param>
// <param name="Signature" type="uint32 "></param>
// <param name="UniqueId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetDisk( /* IN */ InputObject []MSFT_Disk,
	/* IN */ UniqueId string,
	/* IN */ Path string,
	/* IN */ Number uint32,
	/* IN */ PartitionStyle uint16,
	/* IN */ IsReadOnly bool,
	/* IN */ IsOffline bool,
	/* IN */ Signature uint32,
	/* IN */ Guid string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDisk", InputObject, UniqueId, Path, Number, PartitionStyle, IsReadOnly, IsOffline, Signature, Guid)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DedupMode" type="uint32 "></param>
// <param name="DriveLetter" type="byte "></param>
// <param name="FileSystemLabel" type="string "></param>
// <param name="InputObject" type="MSFT_Volume []"></param>
// <param name="NewFileSystemLabel" type="string "></param>
// <param name="Path" type="string "></param>
// <param name="UniqueId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetVolume( /* IN */ InputObject []MSFT_Volume,
	/* IN */ UniqueId string,
	/* IN */ Path string,
	/* IN */ FileSystemLabel string,
	/* IN */ DriveLetter byte,
	/* IN */ NewFileSystemLabel string,
	/* IN */ DedupMode uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetVolume", InputObject, UniqueId, Path, FileSystemLabel, DriveLetter, NewFileSystemLabel, DedupMode)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DiskId" type="string "></param>
// <param name="DiskNumber" type="uint32 "></param>
// <param name="DriveLetter" type="byte "></param>
// <param name="GptType" type="string "></param>
// <param name="InputObject" type="MSFT_Partition []"></param>
// <param name="IsActive" type="bool "></param>
// <param name="IsDAX" type="bool "></param>
// <param name="IsHidden" type="bool "></param>
// <param name="IsOffline" type="bool "></param>
// <param name="IsReadOnly" type="bool "></param>
// <param name="IsShadowCopy" type="bool "></param>
// <param name="MbrType" type="uint16 "></param>
// <param name="NewDriveLetter" type="byte "></param>
// <param name="NoDefaultDriveLetter" type="bool "></param>
// <param name="Offset" type="uint64 "></param>
// <param name="PartitionNumber" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetPartition( /* IN */ InputObject []MSFT_Partition,
	/* IN */ DiskId string,
	/* IN */ Offset uint64,
	/* IN */ DiskNumber uint32,
	/* IN */ PartitionNumber uint32,
	/* IN */ DriveLetter byte,
	/* IN */ NewDriveLetter byte,
	/* IN */ IsOffline bool,
	/* IN */ IsReadOnly bool,
	/* IN */ NoDefaultDriveLetter bool,
	/* IN */ IsActive bool,
	/* IN */ IsHidden bool,
	/* IN */ IsShadowCopy bool,
	/* IN */ IsDAX bool,
	/* IN */ MbrType uint16,
	/* IN */ GptType string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetPartition", InputObject, DiskId, Offset, DiskNumber, PartitionNumber, DriveLetter, NewDriveLetter, IsOffline, IsReadOnly, NoDefaultDriveLetter, IsActive, IsHidden, IsShadowCopy, IsDAX, MbrType, GptType)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Description" type="string "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="InputObject" type="MSFT_PhysicalDisk []"></param>
// <param name="MediaType" type="uint16 "></param>
// <param name="NewFriendlyName" type="string "></param>
// <param name="StorageEnclosureId" type="string "></param>
// <param name="StorageScaleUnitId" type="string "></param>
// <param name="UniqueId" type="string "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetPhysicalDisk( /* IN */ InputObject []MSFT_PhysicalDisk,
	/* IN */ UniqueId string,
	/* IN */ FriendlyName string,
	/* IN */ NewFriendlyName string,
	/* IN */ Description string,
	/* IN */ Usage uint16,
	/* IN */ MediaType uint16,
	/* IN */ StorageEnclosureId string,
	/* IN */ StorageScaleUnitId string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetPhysicalDisk", InputObject, UniqueId, FriendlyName, NewFriendlyName, Description, Usage, MediaType, StorageEnclosureId, StorageScaleUnitId)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="AutoWriteCacheSize" type="bool "></param>
// <param name="ClearOnDeallocate" type="bool "></param>
// <param name="EnclosureAwareDefault" type="bool "></param>
// <param name="FaultDomainAwarenessDefault" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="InputObject" type="MSFT_StoragePool []"></param>
// <param name="IsPowerProtected" type="bool "></param>
// <param name="IsReadOnly" type="bool "></param>
// <param name="MediaTypeDefault" type="uint16 "></param>
// <param name="Name" type="string "></param>
// <param name="NewFriendlyName" type="string "></param>
// <param name="OtherUsageDescription" type="string "></param>
// <param name="ProvisioningTypeDefault" type="uint16 "></param>
// <param name="RepairPolicy" type="uint16 "></param>
// <param name="ResiliencySettingNameDefault" type="string "></param>
// <param name="RetireMissingPhysicalDisks" type="uint16 "></param>
// <param name="ThinProvisioningAlertThresholds" type="uint16 []"></param>
// <param name="UniqueId" type="string "></param>
// <param name="Usage" type="uint16 "></param>
// <param name="WriteCacheSizeDefault" type="uint64 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetStoragePool( /* IN */ InputObject []MSFT_StoragePool,
	/* IN */ UniqueId string,
	/* IN */ Name string,
	/* IN */ FriendlyName string,
	/* IN */ NewFriendlyName string,
	/* IN */ Usage uint16,
	/* IN */ OtherUsageDescription string,
	/* IN */ ProvisioningTypeDefault uint16,
	/* IN */ MediaTypeDefault uint16,
	/* IN */ ResiliencySettingNameDefault string,
	/* IN */ EnclosureAwareDefault bool,
	/* IN */ FaultDomainAwarenessDefault uint16,
	/* IN */ WriteCacheSizeDefault uint64,
	/* IN */ AutoWriteCacheSize bool,
	/* IN */ IsReadOnly bool,
	/* IN */ ClearOnDeallocate bool,
	/* IN */ IsPowerProtected bool,
	/* IN */ RepairPolicy uint16,
	/* IN */ RetireMissingPhysicalDisks uint16,
	/* IN */ ThinProvisioningAlertThresholds []uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetStoragePool", InputObject, UniqueId, Name, FriendlyName, NewFriendlyName, Usage, OtherUsageDescription, ProvisioningTypeDefault, MediaTypeDefault, ResiliencySettingNameDefault, EnclosureAwareDefault, FaultDomainAwarenessDefault, WriteCacheSizeDefault, AutoWriteCacheSize, IsReadOnly, ClearOnDeallocate, IsPowerProtected, RepairPolicy, RetireMissingPhysicalDisks, ThinProvisioningAlertThresholds)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="Access" type="uint16 "></param>
// <param name="AllocationUnitSize" type="uint64 "></param>
// <param name="ColumnIsolation" type="uint16 "></param>
// <param name="FaultDomainAwareness" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="InputObject" type="MSFT_VirtualDisk []"></param>
// <param name="Interleave" type="uint64 "></param>
// <param name="IsManualAttach" type="bool "></param>
// <param name="MediaType" type="uint16 "></param>
// <param name="Name" type="string "></param>
// <param name="NewFriendlyName" type="string "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="NumberOfDataCopies" type="uint16 "></param>
// <param name="NumberOfGroups" type="uint16 "></param>
// <param name="OtherUsageDescription" type="string "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="ResiliencySettingName" type="string "></param>
// <param name="StorageNodeName" type="string "></param>
// <param name="UniqueId" type="string "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetVirtualDisk( /* IN */ InputObject []MSFT_VirtualDisk,
	/* IN */ UniqueId string,
	/* IN */ Name string,
	/* IN */ FriendlyName string,
	/* IN */ NewFriendlyName string,
	/* IN */ Usage uint16,
	/* IN */ OtherUsageDescription string,
	/* IN */ IsManualAttach bool,
	/* IN */ StorageNodeName string,
	/* IN */ Access uint16,
	/* IN */ ProvisioningType uint16,
	/* IN */ AllocationUnitSize uint64,
	/* IN */ MediaType uint16,
	/* IN */ FaultDomainAwareness uint16,
	/* IN */ ColumnIsolation uint16,
	/* IN */ ResiliencySettingName string,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ NumberOfDataCopies uint16,
	/* IN */ NumberOfGroups uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ Interleave uint64) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetVirtualDisk", InputObject, UniqueId, Name, FriendlyName, NewFriendlyName, Usage, OtherUsageDescription, IsManualAttach, StorageNodeName, Access, ProvisioningType, AllocationUnitSize, MediaType, FaultDomainAwareness, ColumnIsolation, ResiliencySettingName, PhysicalDiskRedundancy, NumberOfDataCopies, NumberOfGroups, NumberOfColumns, Interleave)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="AllocationUnitSize" type="uint64 "></param>
// <param name="ColumnIsolation" type="uint16 "></param>
// <param name="Description" type="string "></param>
// <param name="FaultDomainAwareness" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="InputObject" type="MSFT_StorageTier []"></param>
// <param name="Interleave" type="uint64 "></param>
// <param name="MediaType" type="uint16 "></param>
// <param name="NewFriendlyName" type="string "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="NumberOfDataCopies" type="uint16 "></param>
// <param name="NumberOfGroups" type="uint16 "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="ResiliencySettingName" type="string "></param>
// <param name="UniqueId" type="string "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetStorageTier( /* IN */ InputObject []MSFT_StorageTier,
	/* IN */ UniqueId string,
	/* IN */ FriendlyName string,
	/* IN */ NewFriendlyName string,
	/* IN */ ProvisioningType uint16,
	/* IN */ AllocationUnitSize uint64,
	/* IN */ MediaType uint16,
	/* IN */ FaultDomainAwareness uint16,
	/* IN */ ColumnIsolation uint16,
	/* IN */ ResiliencySettingName string,
	/* IN */ Usage uint16,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ NumberOfDataCopies uint16,
	/* IN */ NumberOfGroups uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ Interleave uint64,
	/* IN */ Description string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetStorageTier", InputObject, UniqueId, FriendlyName, NewFriendlyName, ProvisioningType, AllocationUnitSize, MediaType, FaultDomainAwareness, ColumnIsolation, ResiliencySettingName, Usage, PhysicalDiskRedundancy, NumberOfDataCopies, NumberOfGroups, NumberOfColumns, Interleave, Description)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="AutomaticClusteringEnabled" type="bool "></param>
// <param name="Description" type="string "></param>
// <param name="FaultDomainAwarenessDefault" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="InputObject" type="MSFT_StorageSubSystem []"></param>
// <param name="Name" type="string "></param>
// <param name="UniqueId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetStorageSubSystem( /* IN */ InputObject []MSFT_StorageSubSystem,
	/* IN */ UniqueId string,
	/* IN */ Name string,
	/* IN */ FriendlyName string,
	/* IN */ Description string,
	/* IN */ AutomaticClusteringEnabled bool,
	/* IN */ FaultDomainAwarenessDefault uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetStorageSubSystem", InputObject, UniqueId, Name, FriendlyName, Description, AutomaticClusteringEnabled, FaultDomainAwarenessDefault)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="PhysicalDisks" type="MSFT_PhysicalDisk []"></param>
// <param name="StoragePool" type="MSFT_StoragePool "></param>
// <param name="StoragePoolFriendlyName" type="string "></param>
// <param name="StoragePoolName" type="string "></param>
// <param name="StoragePoolUniqueId" type="string "></param>
// <param name="Usage" type="uint16 "></param>
// <param name="VirtualDisk" type="MSFT_VirtualDisk "></param>
// <param name="VirtualDiskFriendlyName" type="string "></param>
// <param name="VirtualDiskName" type="string "></param>
// <param name="VirtualDiskUniqueId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) AddPhysicalDisk( /* IN */ StoragePool MSFT_StoragePool,
	/* IN */ StoragePoolUniqueId string,
	/* IN */ StoragePoolName string,
	/* IN */ StoragePoolFriendlyName string,
	/* IN */ VirtualDisk MSFT_VirtualDisk,
	/* IN */ VirtualDiskUniqueId string,
	/* IN */ VirtualDiskName string,
	/* IN */ VirtualDiskFriendlyName string,
	/* IN */ PhysicalDisks []MSFT_PhysicalDisk,
	/* IN */ Usage uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("AddPhysicalDisk", StoragePool, StoragePoolUniqueId, StoragePoolName, StoragePoolFriendlyName, VirtualDisk, VirtualDiskUniqueId, VirtualDiskName, VirtualDiskFriendlyName, PhysicalDisks, Usage)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="PhysicalDisks" type="MSFT_PhysicalDisk []"></param>
// <param name="StoragePool" type="MSFT_StoragePool "></param>
// <param name="StoragePoolFriendlyName" type="string "></param>
// <param name="StoragePoolName" type="string "></param>
// <param name="StoragePoolUniqueId" type="string "></param>
// <param name="VirtualDisk" type="MSFT_VirtualDisk "></param>
// <param name="VirtualDiskFriendlyName" type="string "></param>
// <param name="VirtualDiskName" type="string "></param>
// <param name="VirtualDiskUniqueId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) RemovePhysicalDisk( /* IN */ StoragePool MSFT_StoragePool,
	/* IN */ StoragePoolUniqueId string,
	/* IN */ StoragePoolName string,
	/* IN */ StoragePoolFriendlyName string,
	/* IN */ VirtualDisk MSFT_VirtualDisk,
	/* IN */ VirtualDiskUniqueId string,
	/* IN */ VirtualDiskName string,
	/* IN */ VirtualDiskFriendlyName string,
	/* IN */ PhysicalDisks []MSFT_PhysicalDisk) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("RemovePhysicalDisk", StoragePool, StoragePoolUniqueId, StoragePoolName, StoragePoolFriendlyName, VirtualDisk, VirtualDiskUniqueId, VirtualDiskName, VirtualDiskFriendlyName, PhysicalDisks)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) LaunchProviderHost( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("LaunchProviderHost")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Description" type="string "></param>
// <param name="EncryptData" type="bool "></param>
// <param name="InputObject" type="MSFT_FileShare []"></param>
// <param name="Name" type="string "></param>
// <param name="UniqueId" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) SetFileShare( /* IN */ InputObject []MSFT_FileShare,
	/* IN */ UniqueId string,
	/* IN */ Name string,
	/* IN */ Description string,
	/* IN */ EncryptData bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetFileShare", InputObject, UniqueId, Name, Description, EncryptData)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="AccessPath" type="string "></param>
// <param name="AllocationUnitSize" type="uint32 "></param>
// <param name="Disk" type="MSFT_Disk "></param>
// <param name="DiskNumber" type="uint32 "></param>
// <param name="DiskPath" type="string "></param>
// <param name="DiskUniqueId" type="string "></param>
// <param name="FileServer" type="MSFT_FileServer "></param>
// <param name="FileSystem" type="uint16 "></param>
// <param name="FriendlyName" type="string "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="ReadCacheSize" type="uint64 "></param>
// <param name="ResiliencySettingName" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Size" type="uint64 "></param>
// <param name="StoragePool" type="MSFT_StoragePool "></param>
// <param name="StoragePoolFriendlyName" type="string "></param>
// <param name="StoragePoolName" type="string "></param>
// <param name="StoragePoolUniqueId" type="string "></param>
// <param name="StorageTiers" type="MSFT_StorageTier []"></param>
// <param name="StorageTierSizes" type="uint64 []"></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedVolume" type="MSFT_Volume []"></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *PS_StorageCmdlets) CreateVolume( /* IN */ StoragePool MSFT_StoragePool,
	/* IN */ StoragePoolUniqueId string,
	/* IN */ StoragePoolName string,
	/* IN */ StoragePoolFriendlyName string,
	/* IN */ Disk MSFT_Disk,
	/* IN */ DiskNumber uint32,
	/* IN */ DiskPath string,
	/* IN */ DiskUniqueId string,
	/* IN */ FriendlyName string,
	/* IN */ Size uint64,
	/* IN */ StorageTiers []MSFT_StorageTier,
	/* IN */ StorageTierSizes []uint64,
	/* IN */ ProvisioningType uint16,
	/* IN */ ResiliencySettingName string,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ FileSystem uint16,
	/* IN */ AccessPath string,
	/* IN */ AllocationUnitSize uint32,
	/* IN */ ReadCacheSize uint64,
	/* IN */ FileServer MSFT_FileServer,
	/* OUT */ CreatedVolume []MSFT_Volume,
	/* OPTIONAL IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateVolume", StoragePool, StoragePoolUniqueId, StoragePoolName, StoragePoolFriendlyName, Disk, DiskNumber, DiskPath, DiskUniqueId, FriendlyName, Size, StorageTiers, StorageTierSizes, ProvisioningType, ResiliencySettingName, PhysicalDiskRedundancy, NumberOfColumns, FileSystem, AccessPath, AllocationUnitSize, ReadCacheSize, FileServer, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Disk" type="MSFT_Disk "></param>
// <param name="PhysicalDisk" type="MSFT_PhysicalDisk "></param>

// <param name="ReturnValue" type="uint32 "></param>
// <param name="StorageReliabilityCounter" type="MSFT_StorageReliabilityCounter "></param>
func (instance *PS_StorageCmdlets) GetStorageReliabilityCounter( /* IN */ PhysicalDisk MSFT_PhysicalDisk,
	/* IN */ Disk MSFT_Disk,
	/* OUT */ StorageReliabilityCounter MSFT_StorageReliabilityCounter) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetStorageReliabilityCounter", PhysicalDisk, Disk)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
