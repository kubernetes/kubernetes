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

// CIM_USBHub struct
type CIM_USBHub struct {
	*CIM_USBDevice

	//
	GangSwitched bool

	//
	NumberOfPorts uint8
}

func NewCIM_USBHubEx1(instance *cim.WmiInstance) (newInstance *CIM_USBHub, err error) {
	tmp, err := NewCIM_USBDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_USBHub{
		CIM_USBDevice: tmp,
	}
	return
}

func NewCIM_USBHubEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_USBHub, err error) {
	tmp, err := NewCIM_USBDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_USBHub{
		CIM_USBDevice: tmp,
	}
	return
}

// SetGangSwitched sets the value of GangSwitched for the instance
func (instance *CIM_USBHub) SetPropertyGangSwitched(value bool) (err error) {
	return instance.SetProperty("GangSwitched", (value))
}

// GetGangSwitched gets the value of GangSwitched for the instance
func (instance *CIM_USBHub) GetPropertyGangSwitched() (value bool, err error) {
	retValue, err := instance.GetProperty("GangSwitched")
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

// SetNumberOfPorts sets the value of NumberOfPorts for the instance
func (instance *CIM_USBHub) SetPropertyNumberOfPorts(value uint8) (err error) {
	return instance.SetProperty("NumberOfPorts", (value))
}

// GetNumberOfPorts gets the value of NumberOfPorts for the instance
func (instance *CIM_USBHub) GetPropertyNumberOfPorts() (value uint8, err error) {
	retValue, err := instance.GetProperty("NumberOfPorts")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}
