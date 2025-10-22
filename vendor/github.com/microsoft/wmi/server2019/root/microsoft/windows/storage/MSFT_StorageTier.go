// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageTier struct
type MSFT_StorageTier struct {
	*MSFT_StorageObject

	//
	AllocatedSize uint64

	//
	AllocationUnitSize uint64

	//
	ColumnIsolation uint16

	//
	Description string

	//
	FaultDomainAwareness uint16

	//
	FootprintOnPool uint64

	//
	FriendlyName string

	//
	Interleave uint64

	//
	MediaType uint16

	//
	NumberOfColumns uint16

	//
	NumberOfDataCopies uint16

	//
	NumberOfGroups uint16

	//
	ParityLayout uint16

	//
	PhysicalDiskRedundancy uint16

	//
	ProvisioningType uint16

	//
	ResiliencySettingName string

	//
	Size uint64

	//
	TierClass uint16

	//
	Usage uint16
}

func NewMSFT_StorageTierEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageTier, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageTier{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_StorageTierEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageTier, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageTier{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetAllocatedSize sets the value of AllocatedSize for the instance
func (instance *MSFT_StorageTier) SetPropertyAllocatedSize(value uint64) (err error) {
	return instance.SetProperty("AllocatedSize", (value))
}

// GetAllocatedSize gets the value of AllocatedSize for the instance
func (instance *MSFT_StorageTier) GetPropertyAllocatedSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("AllocatedSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetAllocationUnitSize sets the value of AllocationUnitSize for the instance
func (instance *MSFT_StorageTier) SetPropertyAllocationUnitSize(value uint64) (err error) {
	return instance.SetProperty("AllocationUnitSize", (value))
}

// GetAllocationUnitSize gets the value of AllocationUnitSize for the instance
func (instance *MSFT_StorageTier) GetPropertyAllocationUnitSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("AllocationUnitSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetColumnIsolation sets the value of ColumnIsolation for the instance
func (instance *MSFT_StorageTier) SetPropertyColumnIsolation(value uint16) (err error) {
	return instance.SetProperty("ColumnIsolation", (value))
}

// GetColumnIsolation gets the value of ColumnIsolation for the instance
func (instance *MSFT_StorageTier) GetPropertyColumnIsolation() (value uint16, err error) {
	retValue, err := instance.GetProperty("ColumnIsolation")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetDescription sets the value of Description for the instance
func (instance *MSFT_StorageTier) SetPropertyDescription(value string) (err error) {
	return instance.SetProperty("Description", (value))
}

// GetDescription gets the value of Description for the instance
func (instance *MSFT_StorageTier) GetPropertyDescription() (value string, err error) {
	retValue, err := instance.GetProperty("Description")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetFaultDomainAwareness sets the value of FaultDomainAwareness for the instance
func (instance *MSFT_StorageTier) SetPropertyFaultDomainAwareness(value uint16) (err error) {
	return instance.SetProperty("FaultDomainAwareness", (value))
}

// GetFaultDomainAwareness gets the value of FaultDomainAwareness for the instance
func (instance *MSFT_StorageTier) GetPropertyFaultDomainAwareness() (value uint16, err error) {
	retValue, err := instance.GetProperty("FaultDomainAwareness")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetFootprintOnPool sets the value of FootprintOnPool for the instance
func (instance *MSFT_StorageTier) SetPropertyFootprintOnPool(value uint64) (err error) {
	return instance.SetProperty("FootprintOnPool", (value))
}

// GetFootprintOnPool gets the value of FootprintOnPool for the instance
func (instance *MSFT_StorageTier) GetPropertyFootprintOnPool() (value uint64, err error) {
	retValue, err := instance.GetProperty("FootprintOnPool")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetFriendlyName sets the value of FriendlyName for the instance
func (instance *MSFT_StorageTier) SetPropertyFriendlyName(value string) (err error) {
	return instance.SetProperty("FriendlyName", (value))
}

// GetFriendlyName gets the value of FriendlyName for the instance
func (instance *MSFT_StorageTier) GetPropertyFriendlyName() (value string, err error) {
	retValue, err := instance.GetProperty("FriendlyName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetInterleave sets the value of Interleave for the instance
func (instance *MSFT_StorageTier) SetPropertyInterleave(value uint64) (err error) {
	return instance.SetProperty("Interleave", (value))
}

// GetInterleave gets the value of Interleave for the instance
func (instance *MSFT_StorageTier) GetPropertyInterleave() (value uint64, err error) {
	retValue, err := instance.GetProperty("Interleave")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetMediaType sets the value of MediaType for the instance
func (instance *MSFT_StorageTier) SetPropertyMediaType(value uint16) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *MSFT_StorageTier) GetPropertyMediaType() (value uint16, err error) {
	retValue, err := instance.GetProperty("MediaType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetNumberOfColumns sets the value of NumberOfColumns for the instance
func (instance *MSFT_StorageTier) SetPropertyNumberOfColumns(value uint16) (err error) {
	return instance.SetProperty("NumberOfColumns", (value))
}

// GetNumberOfColumns gets the value of NumberOfColumns for the instance
func (instance *MSFT_StorageTier) GetPropertyNumberOfColumns() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfColumns")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetNumberOfDataCopies sets the value of NumberOfDataCopies for the instance
func (instance *MSFT_StorageTier) SetPropertyNumberOfDataCopies(value uint16) (err error) {
	return instance.SetProperty("NumberOfDataCopies", (value))
}

// GetNumberOfDataCopies gets the value of NumberOfDataCopies for the instance
func (instance *MSFT_StorageTier) GetPropertyNumberOfDataCopies() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfDataCopies")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetNumberOfGroups sets the value of NumberOfGroups for the instance
func (instance *MSFT_StorageTier) SetPropertyNumberOfGroups(value uint16) (err error) {
	return instance.SetProperty("NumberOfGroups", (value))
}

// GetNumberOfGroups gets the value of NumberOfGroups for the instance
func (instance *MSFT_StorageTier) GetPropertyNumberOfGroups() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfGroups")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetParityLayout sets the value of ParityLayout for the instance
func (instance *MSFT_StorageTier) SetPropertyParityLayout(value uint16) (err error) {
	return instance.SetProperty("ParityLayout", (value))
}

// GetParityLayout gets the value of ParityLayout for the instance
func (instance *MSFT_StorageTier) GetPropertyParityLayout() (value uint16, err error) {
	retValue, err := instance.GetProperty("ParityLayout")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetPhysicalDiskRedundancy sets the value of PhysicalDiskRedundancy for the instance
func (instance *MSFT_StorageTier) SetPropertyPhysicalDiskRedundancy(value uint16) (err error) {
	return instance.SetProperty("PhysicalDiskRedundancy", (value))
}

// GetPhysicalDiskRedundancy gets the value of PhysicalDiskRedundancy for the instance
func (instance *MSFT_StorageTier) GetPropertyPhysicalDiskRedundancy() (value uint16, err error) {
	retValue, err := instance.GetProperty("PhysicalDiskRedundancy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetProvisioningType sets the value of ProvisioningType for the instance
func (instance *MSFT_StorageTier) SetPropertyProvisioningType(value uint16) (err error) {
	return instance.SetProperty("ProvisioningType", (value))
}

// GetProvisioningType gets the value of ProvisioningType for the instance
func (instance *MSFT_StorageTier) GetPropertyProvisioningType() (value uint16, err error) {
	retValue, err := instance.GetProperty("ProvisioningType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetResiliencySettingName sets the value of ResiliencySettingName for the instance
func (instance *MSFT_StorageTier) SetPropertyResiliencySettingName(value string) (err error) {
	return instance.SetProperty("ResiliencySettingName", (value))
}

// GetResiliencySettingName gets the value of ResiliencySettingName for the instance
func (instance *MSFT_StorageTier) GetPropertyResiliencySettingName() (value string, err error) {
	retValue, err := instance.GetProperty("ResiliencySettingName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetSize sets the value of Size for the instance
func (instance *MSFT_StorageTier) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *MSFT_StorageTier) GetPropertySize() (value uint64, err error) {
	retValue, err := instance.GetProperty("Size")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTierClass sets the value of TierClass for the instance
func (instance *MSFT_StorageTier) SetPropertyTierClass(value uint16) (err error) {
	return instance.SetProperty("TierClass", (value))
}

// GetTierClass gets the value of TierClass for the instance
func (instance *MSFT_StorageTier) GetPropertyTierClass() (value uint16, err error) {
	retValue, err := instance.GetProperty("TierClass")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetUsage sets the value of Usage for the instance
func (instance *MSFT_StorageTier) SetPropertyUsage(value uint16) (err error) {
	return instance.SetProperty("Usage", (value))
}

// GetUsage gets the value of Usage for the instance
func (instance *MSFT_StorageTier) GetPropertyUsage() (value uint16, err error) {
	retValue, err := instance.GetProperty("Usage")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="PhysicalExtents" type="MSFT_PhysicalExtent []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageTier) GetPhysicalExtent( /* OUT */ PhysicalExtents []MSFT_PhysicalExtent,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetPhysicalExtent")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageTier) DeleteObject( /* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("DeleteObject", RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>
// <param name="Size" type="uint64 "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="Size" type="uint64 "></param>
func (instance *MSFT_StorageTier) Resize( /* IN/OUT */ Size uint64,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Resize", RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FriendlyName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageTier) SetFriendlyName( /* IN */ FriendlyName string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetFriendlyName", FriendlyName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="AllocationUnitSize" type="uint64 "></param>
// <param name="ColumnIsolation" type="uint16 "></param>
// <param name="FaultDomainAwareness" type="uint16 "></param>
// <param name="Interleave" type="uint64 "></param>
// <param name="MediaType" type="uint16 "></param>
// <param name="NumberOfColumns" type="uint16 "></param>
// <param name="NumberOfDataCopies" type="uint16 "></param>
// <param name="NumberOfGroups" type="uint16 "></param>
// <param name="PhysicalDiskRedundancy" type="uint16 "></param>
// <param name="ProvisioningType" type="uint16 "></param>
// <param name="ResiliencySettingName" type="string "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageTier) SetAttributes( /* IN */ ProvisioningType uint16,
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
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetAttributes", ProvisioningType, AllocationUnitSize, MediaType, FaultDomainAwareness, ColumnIsolation, ResiliencySettingName, Usage, PhysicalDiskRedundancy, NumberOfDataCopies, NumberOfGroups, NumberOfColumns, Interleave)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Description" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageTier) SetDescription( /* IN */ Description string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetDescription", Description)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ResiliencySettingName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SupportedSizes" type="uint64 []"></param>
// <param name="TierSizeDivisor" type="uint64 "></param>
// <param name="TierSizeMax" type="uint64 "></param>
// <param name="TierSizeMin" type="uint64 "></param>
func (instance *MSFT_StorageTier) GetSupportedSize( /* IN */ ResiliencySettingName string,
	/* OUT */ SupportedSizes []uint64,
	/* OUT */ TierSizeMin uint64,
	/* OUT */ TierSizeMax uint64,
	/* OUT */ TierSizeDivisor uint64,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSupportedSize", ResiliencySettingName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>
// <param name="StorageFaultDomains" type="MSFT_StorageFaultDomain []"></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageTier) AddStorageFaultDomain( /* IN */ StorageFaultDomains []MSFT_StorageFaultDomain,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("AddStorageFaultDomain", StorageFaultDomains, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="RunAsJob" type="bool "></param>
// <param name="StorageFaultDomains" type="MSFT_StorageFaultDomain []"></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageTier) RemoveStorageFaultDomain( /* IN */ StorageFaultDomains []MSFT_StorageFaultDomain,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("RemoveStorageFaultDomain", StorageFaultDomains, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
