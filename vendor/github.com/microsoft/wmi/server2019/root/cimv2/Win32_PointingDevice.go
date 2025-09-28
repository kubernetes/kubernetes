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

// Win32_PointingDevice struct
type Win32_PointingDevice struct {
	*CIM_PointingDevice

	//
	DeviceInterface uint16

	//
	DoubleSpeedThreshold uint32

	//
	HardwareType string

	//
	InfFileName string

	//
	InfSection string

	//
	Manufacturer string

	//
	QuadSpeedThreshold uint32

	//
	SampleRate uint32

	//
	Synch uint32
}

func NewWin32_PointingDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_PointingDevice, err error) {
	tmp, err := NewCIM_PointingDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PointingDevice{
		CIM_PointingDevice: tmp,
	}
	return
}

func NewWin32_PointingDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PointingDevice, err error) {
	tmp, err := NewCIM_PointingDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PointingDevice{
		CIM_PointingDevice: tmp,
	}
	return
}

// SetDeviceInterface sets the value of DeviceInterface for the instance
func (instance *Win32_PointingDevice) SetPropertyDeviceInterface(value uint16) (err error) {
	return instance.SetProperty("DeviceInterface", (value))
}

// GetDeviceInterface gets the value of DeviceInterface for the instance
func (instance *Win32_PointingDevice) GetPropertyDeviceInterface() (value uint16, err error) {
	retValue, err := instance.GetProperty("DeviceInterface")
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

// SetDoubleSpeedThreshold sets the value of DoubleSpeedThreshold for the instance
func (instance *Win32_PointingDevice) SetPropertyDoubleSpeedThreshold(value uint32) (err error) {
	return instance.SetProperty("DoubleSpeedThreshold", (value))
}

// GetDoubleSpeedThreshold gets the value of DoubleSpeedThreshold for the instance
func (instance *Win32_PointingDevice) GetPropertyDoubleSpeedThreshold() (value uint32, err error) {
	retValue, err := instance.GetProperty("DoubleSpeedThreshold")
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

// SetHardwareType sets the value of HardwareType for the instance
func (instance *Win32_PointingDevice) SetPropertyHardwareType(value string) (err error) {
	return instance.SetProperty("HardwareType", (value))
}

// GetHardwareType gets the value of HardwareType for the instance
func (instance *Win32_PointingDevice) GetPropertyHardwareType() (value string, err error) {
	retValue, err := instance.GetProperty("HardwareType")
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

// SetInfFileName sets the value of InfFileName for the instance
func (instance *Win32_PointingDevice) SetPropertyInfFileName(value string) (err error) {
	return instance.SetProperty("InfFileName", (value))
}

// GetInfFileName gets the value of InfFileName for the instance
func (instance *Win32_PointingDevice) GetPropertyInfFileName() (value string, err error) {
	retValue, err := instance.GetProperty("InfFileName")
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

// SetInfSection sets the value of InfSection for the instance
func (instance *Win32_PointingDevice) SetPropertyInfSection(value string) (err error) {
	return instance.SetProperty("InfSection", (value))
}

// GetInfSection gets the value of InfSection for the instance
func (instance *Win32_PointingDevice) GetPropertyInfSection() (value string, err error) {
	retValue, err := instance.GetProperty("InfSection")
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

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *Win32_PointingDevice) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_PointingDevice) GetPropertyManufacturer() (value string, err error) {
	retValue, err := instance.GetProperty("Manufacturer")
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

// SetQuadSpeedThreshold sets the value of QuadSpeedThreshold for the instance
func (instance *Win32_PointingDevice) SetPropertyQuadSpeedThreshold(value uint32) (err error) {
	return instance.SetProperty("QuadSpeedThreshold", (value))
}

// GetQuadSpeedThreshold gets the value of QuadSpeedThreshold for the instance
func (instance *Win32_PointingDevice) GetPropertyQuadSpeedThreshold() (value uint32, err error) {
	retValue, err := instance.GetProperty("QuadSpeedThreshold")
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

// SetSampleRate sets the value of SampleRate for the instance
func (instance *Win32_PointingDevice) SetPropertySampleRate(value uint32) (err error) {
	return instance.SetProperty("SampleRate", (value))
}

// GetSampleRate gets the value of SampleRate for the instance
func (instance *Win32_PointingDevice) GetPropertySampleRate() (value uint32, err error) {
	retValue, err := instance.GetProperty("SampleRate")
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

// SetSynch sets the value of Synch for the instance
func (instance *Win32_PointingDevice) SetPropertySynch(value uint32) (err error) {
	return instance.SetProperty("Synch", (value))
}

// GetSynch gets the value of Synch for the instance
func (instance *Win32_PointingDevice) GetPropertySynch() (value uint32, err error) {
	retValue, err := instance.GetProperty("Synch")
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
