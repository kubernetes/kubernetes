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

// Win32_ComputerSystemProduct struct
type Win32_ComputerSystemProduct struct {
	*CIM_Product

	//
	UUID string
}

func NewWin32_ComputerSystemProductEx1(instance *cim.WmiInstance) (newInstance *Win32_ComputerSystemProduct, err error) {
	tmp, err := NewCIM_ProductEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystemProduct{
		CIM_Product: tmp,
	}
	return
}

func NewWin32_ComputerSystemProductEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ComputerSystemProduct, err error) {
	tmp, err := NewCIM_ProductEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystemProduct{
		CIM_Product: tmp,
	}
	return
}

// SetUUID sets the value of UUID for the instance
func (instance *Win32_ComputerSystemProduct) SetPropertyUUID(value string) (err error) {
	return instance.SetProperty("UUID", (value))
}

// GetUUID gets the value of UUID for the instance
func (instance *Win32_ComputerSystemProduct) GetPropertyUUID() (value string, err error) {
	retValue, err := instance.GetProperty("UUID")
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
