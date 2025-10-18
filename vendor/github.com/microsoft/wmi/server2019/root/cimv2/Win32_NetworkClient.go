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

// Win32_NetworkClient struct
type Win32_NetworkClient struct {
	*CIM_LogicalElement

	//
	Manufacturer string
}

func NewWin32_NetworkClientEx1(instance *cim.WmiInstance) (newInstance *Win32_NetworkClient, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkClient{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_NetworkClientEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NetworkClient, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkClient{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetManufacturer sets the value of Manufacturer for the instance
func (instance *Win32_NetworkClient) SetPropertyManufacturer(value string) (err error) {
	return instance.SetProperty("Manufacturer", (value))
}

// GetManufacturer gets the value of Manufacturer for the instance
func (instance *Win32_NetworkClient) GetPropertyManufacturer() (value string, err error) {
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
