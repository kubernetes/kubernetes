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

// Win32_OnBoardDevice struct
type Win32_OnBoardDevice struct {
	*CIM_PhysicalComponent

	//
	DeviceType uint16

	//
	Enabled bool
}

func NewWin32_OnBoardDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_OnBoardDevice, err error) {
	tmp, err := NewCIM_PhysicalComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_OnBoardDevice{
		CIM_PhysicalComponent: tmp,
	}
	return
}

func NewWin32_OnBoardDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_OnBoardDevice, err error) {
	tmp, err := NewCIM_PhysicalComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_OnBoardDevice{
		CIM_PhysicalComponent: tmp,
	}
	return
}

// SetDeviceType sets the value of DeviceType for the instance
func (instance *Win32_OnBoardDevice) SetPropertyDeviceType(value uint16) (err error) {
	return instance.SetProperty("DeviceType", (value))
}

// GetDeviceType gets the value of DeviceType for the instance
func (instance *Win32_OnBoardDevice) GetPropertyDeviceType() (value uint16, err error) {
	retValue, err := instance.GetProperty("DeviceType")
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

// SetEnabled sets the value of Enabled for the instance
func (instance *Win32_OnBoardDevice) SetPropertyEnabled(value bool) (err error) {
	return instance.SetProperty("Enabled", (value))
}

// GetEnabled gets the value of Enabled for the instance
func (instance *Win32_OnBoardDevice) GetPropertyEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("Enabled")
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
