// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_PhysicalMemory struct
type Win32_PhysicalMemory struct {
	*CIM_PhysicalMemory

	//
	Attributes uint32

	//
	ConfiguredClockSpeed uint32

	//
	ConfiguredVoltage uint32

	//
	DeviceLocator string

	//
	InterleaveDataDepth uint16

	//
	MaxVoltage uint32

	//
	MinVoltage uint32

	//
	SMBIOSMemoryType uint32

	//
	TypeDetail uint16
}

func NewWin32_PhysicalMemoryEx1(instance *cim.WmiInstance) (newInstance *Win32_PhysicalMemory, err error) {
	tmp, err := NewCIM_PhysicalMemoryEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PhysicalMemory{
		CIM_PhysicalMemory: tmp,
	}
	return
}

func NewWin32_PhysicalMemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PhysicalMemory, err error) {
	tmp, err := NewCIM_PhysicalMemoryEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PhysicalMemory{
		CIM_PhysicalMemory: tmp,
	}
	return
}

// SetAttributes sets the value of Attributes for the instance
func (instance *Win32_PhysicalMemory) SetPropertyAttributes(value uint32) (err error) {
	return instance.SetProperty("Attributes", (value))
}

// GetAttributes gets the value of Attributes for the instance
func (instance *Win32_PhysicalMemory) GetPropertyAttributes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Attributes")
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

// SetConfiguredClockSpeed sets the value of ConfiguredClockSpeed for the instance
func (instance *Win32_PhysicalMemory) SetPropertyConfiguredClockSpeed(value uint32) (err error) {
	return instance.SetProperty("ConfiguredClockSpeed", (value))
}

// GetConfiguredClockSpeed gets the value of ConfiguredClockSpeed for the instance
func (instance *Win32_PhysicalMemory) GetPropertyConfiguredClockSpeed() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConfiguredClockSpeed")
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

// SetConfiguredVoltage sets the value of ConfiguredVoltage for the instance
func (instance *Win32_PhysicalMemory) SetPropertyConfiguredVoltage(value uint32) (err error) {
	return instance.SetProperty("ConfiguredVoltage", (value))
}

// GetConfiguredVoltage gets the value of ConfiguredVoltage for the instance
func (instance *Win32_PhysicalMemory) GetPropertyConfiguredVoltage() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConfiguredVoltage")
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

// SetDeviceLocator sets the value of DeviceLocator for the instance
func (instance *Win32_PhysicalMemory) SetPropertyDeviceLocator(value string) (err error) {
	return instance.SetProperty("DeviceLocator", (value))
}

// GetDeviceLocator gets the value of DeviceLocator for the instance
func (instance *Win32_PhysicalMemory) GetPropertyDeviceLocator() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceLocator")
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

// SetInterleaveDataDepth sets the value of InterleaveDataDepth for the instance
func (instance *Win32_PhysicalMemory) SetPropertyInterleaveDataDepth(value uint16) (err error) {
	return instance.SetProperty("InterleaveDataDepth", (value))
}

// GetInterleaveDataDepth gets the value of InterleaveDataDepth for the instance
func (instance *Win32_PhysicalMemory) GetPropertyInterleaveDataDepth() (value uint16, err error) {
	retValue, err := instance.GetProperty("InterleaveDataDepth")
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

// SetMaxVoltage sets the value of MaxVoltage for the instance
func (instance *Win32_PhysicalMemory) SetPropertyMaxVoltage(value uint32) (err error) {
	return instance.SetProperty("MaxVoltage", (value))
}

// GetMaxVoltage gets the value of MaxVoltage for the instance
func (instance *Win32_PhysicalMemory) GetPropertyMaxVoltage() (value uint32, err error) {
	retValue, err := instance.GetProperty("MaxVoltage")
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

// SetMinVoltage sets the value of MinVoltage for the instance
func (instance *Win32_PhysicalMemory) SetPropertyMinVoltage(value uint32) (err error) {
	return instance.SetProperty("MinVoltage", (value))
}

// GetMinVoltage gets the value of MinVoltage for the instance
func (instance *Win32_PhysicalMemory) GetPropertyMinVoltage() (value uint32, err error) {
	retValue, err := instance.GetProperty("MinVoltage")
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

// SetSMBIOSMemoryType sets the value of SMBIOSMemoryType for the instance
func (instance *Win32_PhysicalMemory) SetPropertySMBIOSMemoryType(value uint32) (err error) {
	return instance.SetProperty("SMBIOSMemoryType", (value))
}

// GetSMBIOSMemoryType gets the value of SMBIOSMemoryType for the instance
func (instance *Win32_PhysicalMemory) GetPropertySMBIOSMemoryType() (value uint32, err error) {
	retValue, err := instance.GetProperty("SMBIOSMemoryType")
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

// SetTypeDetail sets the value of TypeDetail for the instance
func (instance *Win32_PhysicalMemory) SetPropertyTypeDetail(value uint16) (err error) {
	return instance.SetProperty("TypeDetail", (value))
}

// GetTypeDetail gets the value of TypeDetail for the instance
func (instance *Win32_PhysicalMemory) GetPropertyTypeDetail() (value uint16, err error) {
	retValue, err := instance.GetProperty("TypeDetail")
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
