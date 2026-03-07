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

// MSFT_StorageEnclosure struct
type MSFT_StorageEnclosure struct {
	*MSFT_StorageFaultDomain

	//
	BusType uint16

	//
	CurrentSensorOperationalStatus []uint16

	//
	DeviceId string

	//
	FanOperationalStatus []uint16

	//
	FirmwareVersion string

	//
	IOControllerOperationalStatus []uint16

	//
	NumberOfSlots uint32

	//
	PowerSupplyOperationalStatus []uint16

	//
	SlotOperationalStatus []uint16

	//
	TemperatureSensorOperationalStatus []uint16

	//
	VoltageSensorOperationalStatus []uint16
}

func NewMSFT_StorageEnclosureEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageEnclosure, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageEnclosure{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}

func NewMSFT_StorageEnclosureEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageEnclosure, err error) {
	tmp, err := NewMSFT_StorageFaultDomainEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageEnclosure{
		MSFT_StorageFaultDomain: tmp,
	}
	return
}

// SetBusType sets the value of BusType for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyBusType(value uint16) (err error) {
	return instance.SetProperty("BusType", (value))
}

// GetBusType gets the value of BusType for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyBusType() (value uint16, err error) {
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

// SetCurrentSensorOperationalStatus sets the value of CurrentSensorOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyCurrentSensorOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("CurrentSensorOperationalStatus", (value))
}

// GetCurrentSensorOperationalStatus gets the value of CurrentSensorOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyCurrentSensorOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("CurrentSensorOperationalStatus")
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

// SetDeviceId sets the value of DeviceId for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyDeviceId(value string) (err error) {
	return instance.SetProperty("DeviceId", (value))
}

// GetDeviceId gets the value of DeviceId for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyDeviceId() (value string, err error) {
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

// SetFanOperationalStatus sets the value of FanOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyFanOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("FanOperationalStatus", (value))
}

// GetFanOperationalStatus gets the value of FanOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyFanOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("FanOperationalStatus")
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

// SetFirmwareVersion sets the value of FirmwareVersion for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyFirmwareVersion(value string) (err error) {
	return instance.SetProperty("FirmwareVersion", (value))
}

// GetFirmwareVersion gets the value of FirmwareVersion for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyFirmwareVersion() (value string, err error) {
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

// SetIOControllerOperationalStatus sets the value of IOControllerOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyIOControllerOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("IOControllerOperationalStatus", (value))
}

// GetIOControllerOperationalStatus gets the value of IOControllerOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyIOControllerOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("IOControllerOperationalStatus")
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

// SetNumberOfSlots sets the value of NumberOfSlots for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyNumberOfSlots(value uint32) (err error) {
	return instance.SetProperty("NumberOfSlots", (value))
}

// GetNumberOfSlots gets the value of NumberOfSlots for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyNumberOfSlots() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumberOfSlots")
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

// SetPowerSupplyOperationalStatus sets the value of PowerSupplyOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyPowerSupplyOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("PowerSupplyOperationalStatus", (value))
}

// GetPowerSupplyOperationalStatus gets the value of PowerSupplyOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyPowerSupplyOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("PowerSupplyOperationalStatus")
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

// SetSlotOperationalStatus sets the value of SlotOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) SetPropertySlotOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("SlotOperationalStatus", (value))
}

// GetSlotOperationalStatus gets the value of SlotOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) GetPropertySlotOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("SlotOperationalStatus")
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

// SetTemperatureSensorOperationalStatus sets the value of TemperatureSensorOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyTemperatureSensorOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("TemperatureSensorOperationalStatus", (value))
}

// GetTemperatureSensorOperationalStatus gets the value of TemperatureSensorOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyTemperatureSensorOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("TemperatureSensorOperationalStatus")
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

// SetVoltageSensorOperationalStatus sets the value of VoltageSensorOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) SetPropertyVoltageSensorOperationalStatus(value []uint16) (err error) {
	return instance.SetProperty("VoltageSensorOperationalStatus", (value))
}

// GetVoltageSensorOperationalStatus gets the value of VoltageSensorOperationalStatus for the instance
func (instance *MSFT_StorageEnclosure) GetPropertyVoltageSensorOperationalStatus() (value []uint16, err error) {
	retValue, err := instance.GetProperty("VoltageSensorOperationalStatus")
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

//

// <param name="Enable" type="bool "></param>
// <param name="SlotNumbers" type="uint32 []"></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageEnclosure) IdentifyElement( /* IN */ Enable bool,
	/* IN */ SlotNumbers []uint32,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("IdentifyElement", Enable, SlotNumbers)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="PageNumber" type="uint16 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
// <param name="VendorData" type="string "></param>
func (instance *MSFT_StorageEnclosure) GetVendorData( /* IN */ PageNumber uint16,
	/* OUT */ VendorData string,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetVendorData", PageNumber)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="EnableMaintenanceMode" type="bool "></param>
// <param name="IgnoreDetachedVirtualDisks" type="bool "></param>
// <param name="Manufacturer" type="string "></param>
// <param name="Model" type="string "></param>
// <param name="Timeout" type="uint32 "></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageEnclosure) Maintenance( /* IN */ EnableMaintenanceMode bool,
	/* IN */ Timeout uint32,
	/* IN */ Model string,
	/* IN */ Manufacturer string,
	/* IN */ IgnoreDetachedVirtualDisks bool,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("Maintenance", EnableMaintenanceMode, Timeout, Model, Manufacturer, IgnoreDetachedVirtualDisks)
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

//

// <param name="Off" type="bool "></param>
// <param name="SlotNumbers" type="uint32 []"></param>

// <param name="ExtendedStatus" type="MSFT_StorageExtendedStatus "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *MSFT_StorageEnclosure) PowerElement( /* IN */ Off bool,
	/* IN */ SlotNumbers []uint32,
	/* OUT */ ExtendedStatus MSFT_StorageExtendedStatus) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("PowerElement", Off, SlotNumbers)
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
func (instance *MSFT_StorageEnclosure) GetFirmwareInformation( /* OUT */ SupportsUpdate bool,
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
func (instance *MSFT_StorageEnclosure) UpdateFirmware( /* IN */ ImagePath string,
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
