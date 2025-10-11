// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_PnPDevice struct
type Win32_PnPDevice struct {
	*cim.WmiInstance

	//
	SameElement CIM_LogicalDevice

	//
	SystemElement Win32_PnPEntity
}

func NewWin32_PnPDeviceEx1(instance *cim.WmiInstance) (newInstance *Win32_PnPDevice, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_PnPDevice{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_PnPDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PnPDevice, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PnPDevice{
		WmiInstance: tmp,
	}
	return
}

// SetSameElement sets the value of SameElement for the instance
func (instance *Win32_PnPDevice) SetPropertySameElement(value CIM_LogicalDevice) (err error) {
	return instance.SetProperty("SameElement", (value))
}

// GetSameElement gets the value of SameElement for the instance
func (instance *Win32_PnPDevice) GetPropertySameElement() (value CIM_LogicalDevice, err error) {
	retValue, err := instance.GetProperty("SameElement")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_LogicalDevice)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_LogicalDevice is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_LogicalDevice(valuetmp)

	return
}

// SetSystemElement sets the value of SystemElement for the instance
func (instance *Win32_PnPDevice) SetPropertySystemElement(value Win32_PnPEntity) (err error) {
	return instance.SetProperty("SystemElement", (value))
}

// GetSystemElement gets the value of SystemElement for the instance
func (instance *Win32_PnPDevice) GetPropertySystemElement() (value Win32_PnPEntity, err error) {
	retValue, err := instance.GetProperty("SystemElement")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_PnPEntity)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_PnPEntity is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_PnPEntity(valuetmp)

	return
}
