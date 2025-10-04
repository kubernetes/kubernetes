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

// Win32_SoundDevice struct
type Win32_SoundDevice struct {
	*CIM_LogicalDevice

	//
	DMABufferSize uint16

	//
	Manufacturer string

	//
	MPU401Address uint32

	//
	ProductName string
}

func NewWin32_SoundDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_SoundDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SoundDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

func NewWin32_SoundDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SoundDevice, err error) {
	tmp, err := NewCIM_LogicalDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SoundDevice{
		CIM_LogicalDevice: tmp,
	}
	return
}

// SetDMABufferSize sets the value of DMABufferSize for the instance
func (instance *Win32_SoundDevice) SetPropertyDMABufferSize(value uint16) (err error) {
	return instance.SetProperty("DMABufferSize", (value))
}

// GetDMABufferSize gets the value of DMABufferSize for the instance
func (instance *Win32_SoundDevice) GetPropertyDMABufferSize() (value uint16, err error) {
	retValue, err := instance.GetProperty("DMABufferSize")
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

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *Win32_SoundDevice) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_SoundDevice) GetPropertyManufacturer() (value string, err error) {
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

// SetMPU401Address sets the value of MPU401Address for the instance
func (instance *Win32_SoundDevice) SetPropertyMPU401Address(value uint32) (err error) {
	return instance.SetProperty("MPU401Address", (value))
}

// GetMPU401Address gets the value of MPU401Address for the instance
func (instance *Win32_SoundDevice) GetPropertyMPU401Address() (value uint32, err error) {
	retValue, err := instance.GetProperty("MPU401Address")
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

// SetProductName sets the value of ProductName for the instance
func (instance *Win32_SoundDevice) SetPropertyProductName(value string) (err error) {
	return instance.SetProperty("ProductName", (value))
}

// GetProductName gets the value of ProductName for the instance
func (instance *Win32_SoundDevice) GetPropertyProductName() (value string, err error) {
	retValue, err := instance.GetProperty("ProductName")
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
