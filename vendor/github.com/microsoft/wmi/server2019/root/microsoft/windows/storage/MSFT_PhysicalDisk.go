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

// MSFT_PhysicalDisk struct
type MSFT_PhysicalDisk struct {
	*MSFT_StorageFaultDomain

	//
	AdapterSerialNumber string

	//
	AllocatedSize uint64

	//
	BusType uint16

	//
	CannotPoolReason []uint16

	//
	CanPool bool

	//
	DeviceId string

	//
	EnclosureNumber uint16

	//
	FirmwareVersion string

	//
	IsIndicationEnabled bool

	//
	IsPartial bool

	//
	LogicalSectorSize uint64

	//
	MediaType uint16

	//
	OtherCannotPoolReasonDescription string

	//
	PartNumber string

	//
	PhysicalSectorSize uint64

	//
	Size uint64

	//
	SlotNumber uint16

	//
	SoftwareVersion string

	//
	SpindleSpeed uint32

	//
	StoragePoolUniqueId string

	//
	SupportedUsages []uint16

	//
	UniqueIdFormat uint16

	//
	Usage uint16

	//
	VirtualDiskFootprint uint64
}

func NewMSFT_PhysicalDiskEx1(instance *cim.WmiInstance) (newInstance *MSFT_PhysicalDisk, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_PhysicalDisk{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}

func NewMSFT_PhysicalDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_PhysicalDisk, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_PhysicalDisk{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}

// SetAdapterSerialNumber sets the value of AdapterSerialNumber for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyAdapterSerialNumber(value string) (err error) {
	return instance.SetProperty("AdapterSerialNumber", (value))
}

// GetAdapterSerialNumber gets the value of AdapterSerialNumber for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyAdapterSerialNumber() (value string, err error) {
	retValue, err := instance.GetProperty("AdapterSerialNumber")
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

// SetAllocatedSize sets the value of AllocatedSize for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyAllocatedSize(value uint64) (err error) {
	return instance.SetProperty("AllocatedSize", (value))
}

// GetAllocatedSize gets the value of AllocatedSize for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyAllocatedSize() (value uint64, err error) {
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

// SetBusType sets the value of BusType for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyBusType(value uint16) (err error) {
	return instance.SetProperty("BusType", (value))
}

// GetBusType gets the value of BusType for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyBusType() (value uint16, err error) {
	retValue, err := instance.GetProperty("BusType")
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

// SetCannotPoolReason sets the value of CannotPoolReason for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyCannotPoolReason(value []uint16) (err error) {
	return instance.SetProperty("CannotPoolReason", (value))
}

// GetCannotPoolReason gets the value of CannotPoolReason for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyCannotPoolReason() (value []uint16, err error) {
	retValue, err := instance.GetProperty("CannotPoolReason")
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

// SetCanPool sets the value of CanPool for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyCanPool(value bool) (err error) {
	return instance.SetProperty("CanPool", (value))
}

// GetCanPool gets the value of CanPool for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyCanPool() (value bool, err error) {
	retValue, err := instance.GetProperty("CanPool")
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

// SetDeviceId sets the value of DeviceId for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyDeviceId(value string) (err error) {
	return instance.SetProperty("DeviceId", (value))
}

// GetDeviceId gets the value of DeviceId for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyDeviceId() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceId")
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

// SetEnclosureNumber sets the value of EnclosureNumber for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyEnclosureNumber(value uint16) (err error) {
	return instance.SetProperty("EnclosureNumber", (value))
}

// GetEnclosureNumber gets the value of EnclosureNumber for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyEnclosureNumber() (value uint16, err error) {
	retValue, err := instance.GetProperty("EnclosureNumber")
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

// SetFirmwareVersion sets the value of FirmwareVersion for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyFirmwareVersion(value string) (err error) {
	return instance.SetProperty("FirmwareVersion", (value))
}

// GetFirmwareVersion gets the value of FirmwareVersion for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyFirmwareVersion() (value string, err error) {
	retValue, err := instance.GetProperty("FirmwareVersion")
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

// SetIsIndicationEnabled sets the value of IsIndicationEnabled for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyIsIndicationEnabled(value bool) (err error) {
	return instance.SetProperty("IsIndicationEnabled", (value))
}

// GetIsIndicationEnabled gets the value of IsIndicationEnabled for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyIsIndicationEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IsIndicationEnabled")
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

// SetIsPartial sets the value of IsPartial for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyIsPartial(value bool) (err error) {
	return instance.SetProperty("IsPartial", (value))
}

// GetIsPartial gets the value of IsPartial for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyIsPartial() (value bool, err error) {
	retValue, err := instance.GetProperty("IsPartial")
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
func (instance *MSFT_PhysicalDisk) SetPropertyLogicalSectorSize(value uint64) (err error) {
	return instance.SetProperty("LogicalSectorSize", (value))
}

// GetLogicalSectorSize gets the value of LogicalSectorSize for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyLogicalSectorSize() (value uint64, err error) {
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
func (instance *MSFT_PhysicalDisk) SetPropertyMediaType(value uint16) (err error) {
	return instance.SetProperty("MediaType", (value))
}

// GetMediaType gets the value of MediaType for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyMediaType() (value uint16, err error) {
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

// SetOtherCannotPoolReasonDescription sets the value of OtherCannotPoolReasonDescription for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyOtherCannotPoolReasonDescription(value string) (err error) {
	return instance.SetProperty("OtherCannotPoolReasonDescription", (value))
}

// GetOtherCannotPoolReasonDescription gets the value of OtherCannotPoolReasonDescription for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyOtherCannotPoolReasonDescription() (value string, err error) {
	retValue, err := instance.GetProperty("OtherCannotPoolReasonDescription")
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

// SetPartNumber sets the value of PartNumber for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyPartNumber(value string) (err error) {
	return instance.SetProperty("PartNumber", (value))
}

// GetPartNumber gets the value of PartNumber for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyPartNumber() (value string, err error) {
	retValue, err := instance.GetProperty("PartNumber")
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

// SetPhysicalSectorSize sets the value of PhysicalSectorSize for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyPhysicalSectorSize(value uint64) (err error) {
	return instance.SetProperty("PhysicalSectorSize", (value))
}

// GetPhysicalSectorSize gets the value of PhysicalSectorSize for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyPhysicalSectorSize() (value uint64, err error) {
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

// SetSize sets the value of Size for the instance
func (instance *MSFT_PhysicalDisk) SetPropertySize(value uint64) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *MSFT_PhysicalDisk) GetPropertySize() (value uint64, err error) {
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

// SetSlotNumber sets the value of SlotNumber for the instance
func (instance *MSFT_PhysicalDisk) SetPropertySlotNumber(value uint16) (err error) {
	return instance.SetProperty("SlotNumber", (value))
}

// GetSlotNumber gets the value of SlotNumber for the instance
func (instance *MSFT_PhysicalDisk) GetPropertySlotNumber() (value uint16, err error) {
	retValue, err := instance.GetProperty("SlotNumber")
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

// SetSoftwareVersion sets the value of SoftwareVersion for the instance
func (instance *MSFT_PhysicalDisk) SetPropertySoftwareVersion(value string) (err error) {
	return instance.SetProperty("SoftwareVersion", (value))
}

// GetSoftwareVersion gets the value of SoftwareVersion for the instance
func (instance *MSFT_PhysicalDisk) GetPropertySoftwareVersion() (value string, err error) {
	retValue, err := instance.GetProperty("SoftwareVersion")
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

// SetSpindleSpeed sets the value of SpindleSpeed for the instance
func (instance *MSFT_PhysicalDisk) SetPropertySpindleSpeed(value uint32) (err error) {
	return instance.SetProperty("SpindleSpeed", (value))
}

// GetSpindleSpeed gets the value of SpindleSpeed for the instance
func (instance *MSFT_PhysicalDisk) GetPropertySpindleSpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("SpindleSpeed")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetStoragePoolUniqueId sets the value of StoragePoolUniqueId for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyStoragePoolUniqueId(value string) (err error) {
	return instance.SetProperty("StoragePoolUniqueId", (value))
}

// GetStoragePoolUniqueId gets the value of StoragePoolUniqueId for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyStoragePoolUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("StoragePoolUniqueId")
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

// SetSupportedUsages sets the value of SupportedUsages for the instance
func (instance *MSFT_PhysicalDisk) SetPropertySupportedUsages(value []uint16) (err error) {
	return instance.SetProperty("SupportedUsages", (value))
}

// GetSupportedUsages gets the value of SupportedUsages for the instance
func (instance *MSFT_PhysicalDisk) GetPropertySupportedUsages() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SupportedUsages")
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

// SetUniqueIdFormat sets the value of UniqueIdFormat for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyUniqueIdFormat(value uint16) (err error) {
	return instance.SetProperty("UniqueIdFormat", (value))
}

// GetUniqueIdFormat gets the value of UniqueIdFormat for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyUniqueIdFormat() (value uint16, err error) {
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

// SetUsage sets the value of Usage for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyUsage(value uint16) (err error) {
	return instance.SetProperty("Usage", (value))
}

// GetUsage gets the value of Usage for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyUsage() (value uint16, err error) {
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

// SetVirtualDiskFootprint sets the value of VirtualDiskFootprint for the instance
func (instance *MSFT_PhysicalDisk) SetPropertyVirtualDiskFootprint(value uint64) (err error) {
	return instance.SetProperty("VirtualDiskFootprint", (value))
}

// GetVirtualDiskFootprint gets the value of VirtualDiskFootprint for the instance
func (instance *MSFT_PhysicalDisk) GetPropertyVirtualDiskFootprint() (value uint64, err error) {
	retValue, err := instance.GetProperty("VirtualDiskFootprint")
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
func (instance *MSFT_PhysicalDisk) GetPhysicalExtent( /* OUT */ PhysicalExtents []MSFT_PhysicalExtent,
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

// <param name="EnableIndication" type="bool "></param>
// <param name="EnableMaintenanceMode" type="bool "></param>
// <param name="IgnoreDetachedVirtualDisks" type="bool "></param>
// <param name="Timeout" type="uint32 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PhysicalDisk) Maintenance( /* IN */ EnableIndication bool,
	/* IN */ EnableMaintenanceMode bool,
	/* IN */ Timeout uint32,
	/* IN */ IgnoreDetachedVirtualDisks bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Maintenance", EnableIndication, EnableMaintenanceMode, Timeout, IgnoreDetachedVirtualDisks)
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
func (instance *MSFT_PhysicalDisk) Reset( /* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Reset")
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
func (instance *MSFT_PhysicalDisk) SetFriendlyName( /* IN */ FriendlyName string,
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

// <param name="Description" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PhysicalDisk) SetDescription( /* IN */ Description string,
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

// <param name="Usage" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PhysicalDisk) SetUsage( /* IN */ Usage uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetUsage", Usage)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="MediaType" type="uint16 "></param>
// <param name="StorageEnclosureId" type="string "></param>
// <param name="StorageScaleUnitId" type="string "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PhysicalDisk) SetAttributes( /* IN */ MediaType uint16,
	/* IN */ StorageEnclosureId string,
	/* IN */ StorageScaleUnitId string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("SetAttributes", MediaType, StorageEnclosureId, StorageScaleUnitId)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="IsDeviceCacheEnabled" type="bool "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PhysicalDisk) IsDeviceCacheEnabled( /* OUT */ IsDeviceCacheEnabled bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("IsDeviceCacheEnabled")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="IsPowerProtected" type="bool "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PhysicalDisk) IsPowerProtected( /* OUT */ IsPowerProtected bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("IsPowerProtected")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ActiveSlotNumber" type="uint16 "></param>
// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="FirmwareVersionInSlot" type="string []"></param>
// <param name="IsSlotWritable" type="bool []"></param>
// <param name="NumberOfSlots" type="uint16 "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="SlotNumber" type="uint16 []"></param>
// <param name="SupportsUpdate" type="bool "></param>
func (instance *MSFT_PhysicalDisk) GetFirmwareInformation( /* OUT */ SupportsUpdate bool,
	/* OUT */ NumberOfSlots uint16,
	/* OUT */ ActiveSlotNumber uint16,
	/* OUT */ SlotNumber []uint16,
	/* OUT */ IsSlotWritable []bool,
	/* OUT */ FirmwareVersionInSlot []string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetFirmwareInformation")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="ImagePath" type="string "></param>
// <param name="SlotNumber" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_PhysicalDisk) UpdateFirmware( /* IN */ ImagePath string,
	/* IN */ SlotNumber uint16,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("UpdateFirmware", ImagePath, SlotNumber)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}
