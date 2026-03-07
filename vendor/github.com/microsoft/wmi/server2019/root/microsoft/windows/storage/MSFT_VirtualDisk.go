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

// MSFT_VirtualDisk struct
type MSFT_VirtualDisk struct {
	*MSFT_StorageObject

	//
	Access uint16

	//
	AllocatedSize uint64

	//
	AllocationUnitSize uint64

	//
	ColumnIsolation uint16

	//
	DetachedReason uint16

	//
	FaultDomainAwareness uint16

	//
	FootprintOnPool uint64

	//
	FriendlyName string

	//
	HealthStatus uint16

	//
	Interleave uint64

	//
	IsDeduplicationEnabled bool

	//
	IsEnclosureAware bool

	//
	IsManualAttach bool

	//
	IsSnapshot bool

	//
	IsTiered bool

	//
	LogicalSectorSize uint64

	//
	MediaType uint16

	//
	Name string

	//
	NameFormat uint16

	//
	NumberOfAvailableCopies uint16

	//
	NumberOfColumns uint16

	//
	NumberOfDataCopies uint16

	//
	NumberOfGroups uint16

	//
	OperationalStatus []uint16

	//
	OtherOperationalStatusDescription string

	//
	OtherUsageDescription string

	//
	ParityLayout uint16

	//
	PhysicalDiskRedundancy uint16

	//
	PhysicalSectorSize uint64

	//
	ProvisioningType uint16

	//
	ReadCacheSize uint64

	//
	RequestNoSinglePointOfFailure bool

	//
	ResiliencySettingName string

	//
	Size uint64

	//
	UniqueIdFormat uint16

	//
	UniqueIdFormatDescription string

	//
	Usage uint16

	//
	WriteCacheSize uint64
}

func NewMSFT_VirtualDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_VirtualDisk, err error) {
	tmp, err := NewMSFT_StorageObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDisk{
		MSFT_StorageObject: tmp,
	}
	return
}

func NewMSFT_VirtualDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_VirtualDisk, err error) {
	tmp, err := NewMSFT_StorageObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_VirtualDisk{
		MSFT_StorageObject: tmp,
	}
	return
}

// SetAccess sets the value of Access for the instance
func (instance *MSFT_VirtualDisk) SetPropertyAccess(value uint16) (err error) {
	return instance.SetProperty("Access", (value))
}

// GetAccess gets the value of Access for the instance
func (instance *MSFT_VirtualDisk) GetPropertyAccess() (value uint16, err error) {
	retValue, err := instance.GetProperty("Access")
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

// SetAllocatedSize sets the value of AllocatedSize for the instance
func (instance *MSFT_VirtualDisk) SetPropertyAllocatedSize(value uint64) (err error) {
	return instance.SetProperty("AllocatedSize", (value))
}

// GetAllocatedSize gets the value of AllocatedSize for the instance
func (instance *MSFT_VirtualDisk) GetPropertyAllocatedSize() (value uint64, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertyAllocationUnitSize(value uint64) (err error) {
	return instance.SetProperty("AllocationUnitSize", (value))
}

// GetAllocationUnitSize gets the value of AllocationUnitSize for the instance
func (instance *MSFT_VirtualDisk) GetPropertyAllocationUnitSize() (value uint64, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertyColumnIsolation(value uint16) (err error) {
	return instance.SetProperty("ColumnIsolation", (value))
}

// GetColumnIsolation gets the value of ColumnIsolation for the instance
func (instance *MSFT_VirtualDisk) GetPropertyColumnIsolation() (value uint16, err error) {
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

// SetDetachedReason sets the value of DetachedReason for the instance
func (instance *MSFT_VirtualDisk) SetPropertyDetachedReason(value uint16) (err error) {
	return instance.SetProperty("DetachedReason", (value))
}

// GetDetachedReason gets the value of DetachedReason for the instance
func (instance *MSFT_VirtualDisk) GetPropertyDetachedReason() (value uint16, err error) {
	retValue, err := instance.GetProperty("DetachedReason")
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

// SetFaultDomainAwareness sets the value of FaultDomainAwareness for the instance
func (instance *MSFT_VirtualDisk) SetPropertyFaultDomainAwareness(value uint16) (err error) {
	return instance.SetProperty("FaultDomainAwareness", (value))
}

// GetFaultDomainAwareness gets the value of FaultDomainAwareness for the instance
func (instance *MSFT_VirtualDisk) GetPropertyFaultDomainAwareness() (value uint16, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertyFootprintOnPool(value uint64) (err error) {
	return instance.SetProperty("FootprintOnPool", (value))
}

// GetFootprintOnPool gets the value of FootprintOnPool for the instance
func (instance *MSFT_VirtualDisk) GetPropertyFootprintOnPool() (value uint64, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertyFriendlyName(value string) (err error) {
	return instance.SetProperty("FriendlyName", (value))
}

// GetFriendlyName gets the value of FriendlyName for the instance
func (instance *MSFT_VirtualDisk) GetPropertyFriendlyName() (value string, err error) {
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

// SetHealthStatus sets the value of HealthStatus for the instance
func (instance *MSFT_VirtualDisk) SetPropertyHealthStatus(value uint16) (err error) {
	return instance.SetProperty("HealthStatus", (value))
}

// GetHealthStatus gets the value of HealthStatus for the instance
func (instance *MSFT_VirtualDisk) GetPropertyHealthStatus() (value uint16, err error) {
	retValue, err := instance.GetProperty("HealthStatus")
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

// SetInterleave sets the value of Interleave for the instance
func (instance *MSFT_VirtualDisk) SetPropertyInterleave(value uint64) (err error) {
	return instance.SetProperty("Interleave", (value))
}

// GetInterleave gets the value of Interleave for the instance
func (instance *MSFT_VirtualDisk) GetPropertyInterleave() (value uint64, err error) {
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

// SetIsDeduplicationEnabled sets the value of IsDeduplicationEnabled for the instance
func (instance *MSFT_VirtualDisk) SetPropertyIsDeduplicationEnabled(value bool) (err error) {
	return instance.SetProperty("IsDeduplicationEnabled", (value))
}

// GetIsDeduplicationEnabled gets the value of IsDeduplicationEnabled for the instance
func (instance *MSFT_VirtualDisk) GetPropertyIsDeduplicationEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IsDeduplicationEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsEnclosureAware sets the value of IsEnclosureAware for the instance
func (instance *MSFT_VirtualDisk) SetPropertyIsEnclosureAware(value bool) (err error) {
	return instance.SetProperty("IsEnclosureAware", (value))
}

// GetIsEnclosureAware gets the value of IsEnclosureAware for the instance
func (instance *MSFT_VirtualDisk) GetPropertyIsEnclosureAware() (value bool, err error) {
	retValue, err := instance.GetProperty("IsEnclosureAware")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsManualAttach sets the value of IsManualAttach for the instance
func (instance *MSFT_VirtualDisk) SetPropertyIsManualAttach(value bool) (err error) {
	return instance.SetProperty("IsManualAttach", (value))
}

// GetIsManualAttach gets the value of IsManualAttach for the instance
func (instance *MSFT_VirtualDisk) GetPropertyIsManualAttach() (value bool, err error) {
	retValue, err := instance.GetProperty("IsManualAttach")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsSnapshot sets the value of IsSnapshot for the instance
func (instance *MSFT_VirtualDisk) SetPropertyIsSnapshot(value bool) (err error) {
	return instance.SetProperty("IsSnapshot", (value))
}

// GetIsSnapshot gets the value of IsSnapshot for the instance
func (instance *MSFT_VirtualDisk) GetPropertyIsSnapshot() (value bool, err error) {
	retValue, err := instance.GetProperty("IsSnapshot")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIsTiered sets the value of IsTiered for the instance
func (instance *MSFT_VirtualDisk) SetPropertyIsTiered(value bool) (err error) {
	return instance.SetProperty("IsTiered", (value))
}

// GetIsTiered gets the value of IsTiered for the instance
func (instance *MSFT_VirtualDisk) GetPropertyIsTiered() (value bool, err error) {
	retValue, err := instance.GetProperty("IsTiered")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetLogicalSectorSize sets the value of LogicalSectorSize for the instance
func (instance *MSFT_VirtualDisk) SetPropertyLogicalSectorSize(value uint64) (err error) {
	return instance.SetProperty("LogicalSectorSize", (value))
}

// GetLogicalSectorSize gets the value of LogicalSectorSize for the instance
func (instance *MSFT_VirtualDisk) GetPropertyLogicalSectorSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("LogicalSectorSize")
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
func (instance *MSFT_VirtualDisk) SetPropertyMediaType(value uint16) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *MSFT_VirtualDisk) GetPropertyMediaType() (value uint16, err error) {
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

// SetName sets the value of Name for the instance
func (instance *MSFT_VirtualDisk) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *MSFT_VirtualDisk) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetNameFormat sets the value of NameFormat for the instance
func (instance *MSFT_VirtualDisk) SetPropertyNameFormat(value uint16) (err error) {
	return instance.SetProperty("NameFormat", (value))
}

// GetNameFormat gets the value of NameFormat for the instance
func (instance *MSFT_VirtualDisk) GetPropertyNameFormat() (value uint16, err error) {
	retValue, err := instance.GetProperty("NameFormat")
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

// SetNumberOfAvailableCopies sets the value of NumberOfAvailableCopies for the instance
func (instance *MSFT_VirtualDisk) SetPropertyNumberOfAvailableCopies(value uint16) (err error) {
	return instance.SetProperty("NumberOfAvailableCopies", (value))
}

// GetNumberOfAvailableCopies gets the value of NumberOfAvailableCopies for the instance
func (instance *MSFT_VirtualDisk) GetPropertyNumberOfAvailableCopies() (value uint16, err error) {
	retValue, err := instance.GetProperty("NumberOfAvailableCopies")
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
func (instance *MSFT_VirtualDisk) SetPropertyNumberOfColumns(value uint16) (err error) {
	return instance.SetProperty("NumberOfColumns", (value))
}

// GetNumberOfColumns gets the value of NumberOfColumns for the instance
func (instance *MSFT_VirtualDisk) GetPropertyNumberOfColumns() (value uint16, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertyNumberOfDataCopies(value uint16) (err error) {
	return instance.SetProperty("NumberOfDataCopies", (value))
}

// GetNumberOfDataCopies gets the value of NumberOfDataCopies for the instance
func (instance *MSFT_VirtualDisk) GetPropertyNumberOfDataCopies() (value uint16, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertyNumberOfGroups(value uint16) (err error) {
	return instance.SetProperty("NumberOfGroups", (value))
}

// GetNumberOfGroups gets the value of NumberOfGroups for the instance
func (instance *MSFT_VirtualDisk) GetPropertyNumberOfGroups() (value uint16, err error) {
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

// SetOperationalStatus sets the value of OperationalStatus for the instance
func (instance *MSFT_VirtualDisk) SetPropertyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("OperationalStatus", (value))
}

// GetOperationalStatus gets the value of OperationalStatus for the instance
func (instance *MSFT_VirtualDisk) GetPropertyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("OperationalStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetOtherOperationalStatusDescription sets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_VirtualDisk) SetPropertyOtherOperationalStatusDescription(value string) (err error) {
	return instance.SetProperty("OtherOperationalStatusDescription", (value))
}

// GetOtherOperationalStatusDescription gets the value of OtherOperationalStatusDescription for the instance
func (instance *MSFT_VirtualDisk) GetPropertyOtherOperationalStatusDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherOperationalStatusDescription")
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

// SetOtherUsageDescription sets the value of OtherUsageDescription for the instance
func (instance *MSFT_VirtualDisk) SetPropertyOtherUsageDescription(value string) (err error) {
	return instance.SetProperty("OtherUsageDescription", (value))
}

// GetOtherUsageDescription gets the value of OtherUsageDescription for the instance
func (instance *MSFT_VirtualDisk) GetPropertyOtherUsageDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherUsageDescription")
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

// SetParityLayout sets the value of ParityLayout for the instance
func (instance *MSFT_VirtualDisk) SetPropertyParityLayout(value uint16) (err error) {
	return instance.SetProperty("ParityLayout", (value))
}

// GetParityLayout gets the value of ParityLayout for the instance
func (instance *MSFT_VirtualDisk) GetPropertyParityLayout() (value uint16, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertyPhysicalDiskRedundancy(value uint16) (err error) {
	return instance.SetProperty("PhysicalDiskRedundancy", (value))
}

// GetPhysicalDiskRedundancy gets the value of PhysicalDiskRedundancy for the instance
func (instance *MSFT_VirtualDisk) GetPropertyPhysicalDiskRedundancy() (value uint16, err error) {
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

// SetPhysicalSectorSize sets the value of PhysicalSectorSize for the instance
func (instance *MSFT_VirtualDisk) SetPropertyPhysicalSectorSize(value uint64) (err error) {
	return instance.SetProperty("PhysicalSectorSize", (value))
}

// GetPhysicalSectorSize gets the value of PhysicalSectorSize for the instance
func (instance *MSFT_VirtualDisk) GetPropertyPhysicalSectorSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("PhysicalSectorSize")
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

// SetProvisioningType sets the value of ProvisioningType for the instance
func (instance *MSFT_VirtualDisk) SetPropertyProvisioningType(value uint16) (err error) {
	return instance.SetProperty("ProvisioningType", (value))
}

// GetProvisioningType gets the value of ProvisioningType for the instance
func (instance *MSFT_VirtualDisk) GetPropertyProvisioningType() (value uint16, err error) {
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

// SetReadCacheSize sets the value of ReadCacheSize for the instance
func (instance *MSFT_VirtualDisk) SetPropertyReadCacheSize(value uint64) (err error) {
	return instance.SetProperty("ReadCacheSize", (value))
}

// GetReadCacheSize gets the value of ReadCacheSize for the instance
func (instance *MSFT_VirtualDisk) GetPropertyReadCacheSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadCacheSize")
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

// SetRequestNoSinglePointOfFailure sets the value of RequestNoSinglePointOfFailure for the instance
func (instance *MSFT_VirtualDisk) SetPropertyRequestNoSinglePointOfFailure(value bool) (err error) {
	return instance.SetProperty("RequestNoSinglePointOfFailure", (value))
}

// GetRequestNoSinglePointOfFailure gets the value of RequestNoSinglePointOfFailure for the instance
func (instance *MSFT_VirtualDisk) GetPropertyRequestNoSinglePointOfFailure() (value bool, err error) {
	retValue, err := instance.GetProperty("RequestNoSinglePointOfFailure")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetResiliencySettingName sets the value of ResiliencySettingName for the instance
func (instance *MSFT_VirtualDisk) SetPropertyResiliencySettingName(value string) (err error) {
	return instance.SetProperty("ResiliencySettingName", (value))
}

// GetResiliencySettingName gets the value of ResiliencySettingName for the instance
func (instance *MSFT_VirtualDisk) GetPropertyResiliencySettingName() (value string, err error) {
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
func (instance *MSFT_VirtualDisk) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *MSFT_VirtualDisk) GetPropertySize() (value uint64, err error) {
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

// SetUniqueIdFormat sets the value of UniqueIdFormat for the instance
func (instance *MSFT_VirtualDisk) SetPropertyUniqueIdFormat(value uint16) (err error) {
	return instance.SetProperty("UniqueIdFormat", (value))
}

// GetUniqueIdFormat gets the value of UniqueIdFormat for the instance
func (instance *MSFT_VirtualDisk) GetPropertyUniqueIdFormat() (value uint16, err error) {
	retValue, err := instance.GetProperty("UniqueIdFormat")
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

// SetUniqueIdFormatDescription sets the value of UniqueIdFormatDescription for the instance
func (instance *MSFT_VirtualDisk) SetPropertyUniqueIdFormatDescription(value string) (err error) {
	return instance.SetProperty("UniqueIdFormatDescription", (value))
}

// GetUniqueIdFormatDescription gets the value of UniqueIdFormatDescription for the instance
func (instance *MSFT_VirtualDisk) GetPropertyUniqueIdFormatDescription() (value string, err error) {
	retValue, err := instance.GetProperty("UniqueIdFormatDescription")
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

// SetUsage sets the value of Usage for the instance
func (instance *MSFT_VirtualDisk) SetPropertyUsage(value uint16) (err error) {
	return instance.SetProperty("Usage", (value))
}

// GetUsage gets the value of Usage for the instance
func (instance *MSFT_VirtualDisk) GetPropertyUsage() (value uint16, err error) {
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

// SetWriteCacheSize sets the value of WriteCacheSize for the instance
func (instance *MSFT_VirtualDisk) SetPropertyWriteCacheSize(value uint64) (err error) {
	return instance.SetProperty("WriteCacheSize", (value))
}

// GetWriteCacheSize gets the value of WriteCacheSize for the instance
func (instance *MSFT_VirtualDisk) GetPropertyWriteCacheSize() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteCacheSize")
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

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="PhysicalExtents" type="MSFT_PhysicalExtent []"></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) GetPhysicalExtent( /* OUT */ PhysicalExtents []MSFT_PhysicalExtent,
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
func (instance *MSFT_VirtualDisk) DeleteObject( /* IN */ RunAsJob bool,
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

// <param name="HostType" type="uint16 "></param>
// <param name="InitiatorAddress" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="TargetPortAddresses" type="string []"></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) Show( /* IN */ TargetPortAddresses []string,
	/* IN */ InitiatorAddress string,
	/* IN */ HostType uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Show", TargetPortAddresses, InitiatorAddress, HostType, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="InitiatorAddress" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="TargetPortAddresses" type="string []"></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) Hide( /* IN */ TargetPortAddresses []string,
	/* IN */ InitiatorAddress string,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Hide", TargetPortAddresses, InitiatorAddress, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FriendlyName" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="TargetStoragePoolName" type="string "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedVirtualDisk" type="MSFT_VirtualDisk "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) CreateSnapshot( /* IN */ FriendlyName string,
	/* IN */ TargetStoragePoolName string,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ CreatedVirtualDisk MSFT_VirtualDisk,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateSnapshot", FriendlyName, TargetStoragePoolName, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="FriendlyName" type="string "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="TargetStoragePoolName" type="string "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="CreatedVirtualDisk" type="MSFT_VirtualDisk "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) CreateClone( /* IN */ FriendlyName string,
	/* IN */ TargetStoragePoolName string,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ CreatedVirtualDisk MSFT_VirtualDisk,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateClone", FriendlyName, TargetStoragePoolName, RunAsJob)
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
func (instance *MSFT_VirtualDisk) Resize( /* IN/OUT */ Size uint64,
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

// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) Repair( /* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Repair", RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SecurityDescriptor" type="string "></param>
func (instance *MSFT_VirtualDisk) GetSecurityDescriptor( /* OUT */ SecurityDescriptor string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSecurityDescriptor")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="SecurityDescriptor" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) SetSecurityDescriptor( /* IN */ SecurityDescriptor string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetSecurityDescriptor", SecurityDescriptor)
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
func (instance *MSFT_VirtualDisk) SetFriendlyName( /* IN */ FriendlyName string,
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

// <param name="OtherUsageDescription" type="string "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) SetUsage( /* IN */ Usage uint16,
	/* IN */ OtherUsageDescription string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetUsage", Usage, OtherUsageDescription)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Access" type="uint16 "></param>
// <param name="IsManualAttach" type="bool "></param>
// <param name="StorageNodeName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) SetAttributes( /* IN */ IsManualAttach bool,
	/* IN */ StorageNodeName string,
	/* IN */ Access uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetAttributes", IsManualAttach, StorageNodeName, Access)
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

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) SetProperties( /* IN */ ProvisioningType uint16,
	/* IN */ AllocationUnitSize uint64,
	/* IN */ MediaType uint16,
	/* IN */ FaultDomainAwareness uint16,
	/* IN */ ColumnIsolation uint16,
	/* IN */ ResiliencySettingName string,
	/* IN */ PhysicalDiskRedundancy uint16,
	/* IN */ NumberOfDataCopies uint16,
	/* IN */ NumberOfGroups uint16,
	/* IN */ NumberOfColumns uint16,
	/* IN */ Interleave uint64,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetProperties", ProvisioningType, AllocationUnitSize, MediaType, FaultDomainAwareness, ColumnIsolation, ResiliencySettingName, PhysicalDiskRedundancy, NumberOfDataCopies, NumberOfGroups, NumberOfColumns, Interleave)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="StorageNodeName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) Attach( /* IN */ StorageNodeName string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Attach", StorageNodeName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="StorageNodeName" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) Detach( /* IN */ StorageNodeName string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Detach", StorageNodeName)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="PhysicalDisks" type="MSFT_PhysicalDisk []"></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="Usage" type="uint16 "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) AddPhysicalDisk( /* IN */ PhysicalDisks []MSFT_PhysicalDisk,
	/* IN */ Usage uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("AddPhysicalDisk", PhysicalDisks, Usage, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="PhysicalDisks" type="MSFT_PhysicalDisk []"></param>
// <param name="RunAsJob" type="bool "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) RemovePhysicalDisk( /* IN */ PhysicalDisks []MSFT_PhysicalDisk,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("RemovePhysicalDisk", PhysicalDisks, RunAsJob)
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
func (instance *MSFT_VirtualDisk) AddStorageFaultDomain( /* IN */ StorageFaultDomains []MSFT_StorageFaultDomain,
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
func (instance *MSFT_VirtualDisk) RemoveStorageFaultDomain( /* IN */ StorageFaultDomains []MSFT_StorageFaultDomain,
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

//

// <param name="FriendlyName" type="string "></param>
// <param name="RecoveryPointObjective" type="uint16 "></param>
// <param name="ReplicationSettings" type="MSFT_ReplicationSettings "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="SyncType" type="uint16 "></param>
// <param name="TargetStoragePoolObjectId" type="string "></param>
// <param name="TargetStorageSubsystem" type="MSFT_ReplicaPeer "></param>
// <param name="TargetVirtualDiskObjectId" type="string "></param>

// <param name="CreatedReplicaPeer" type="MSFT_ReplicaPeer "></param>
// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) CreateReplica( /* IN */ FriendlyName string,
	/* IN */ TargetStorageSubsystem MSFT_ReplicaPeer,
	/* IN */ TargetVirtualDiskObjectId string,
	/* IN */ TargetStoragePoolObjectId string,
	/* IN */ RecoveryPointObjective uint16,
	/* IN */ ReplicationSettings MSFT_ReplicationSettings,
	/* IN */ SyncType uint16,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedReplicaPeer MSFT_ReplicaPeer,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("CreateReplica", FriendlyName, TargetStorageSubsystem, TargetVirtualDiskObjectId, TargetStoragePoolObjectId, RecoveryPointObjective, ReplicationSettings, SyncType, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Operation" type="uint16 "></param>
// <param name="RunAsJob" type="bool "></param>
// <param name="VirtualDiskReplicaPeer" type="MSFT_ReplicaPeer "></param>

// <param name="CreatedStorageJob" type="MSFT_StorageJob "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_VirtualDisk) SetReplicationRelationship( /* IN */ Operation uint16,
	/* IN */ VirtualDiskReplicaPeer MSFT_ReplicaPeer,
	/* IN */ RunAsJob bool,
	/* OUT */ CreatedStorageJob MSFT_StorageJob,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetReplicationRelationship", Operation, VirtualDiskReplicaPeer, RunAsJob)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
