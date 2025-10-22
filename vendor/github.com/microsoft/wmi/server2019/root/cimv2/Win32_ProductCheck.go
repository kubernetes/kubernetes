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

// Win32_ProductCheck struct
type Win32_ProductCheck struct {
	*cim.WmiInstance

	//
	Check CIM_Check

	//
	Product Win32_Product
}

func NewWin32_ProductCheckEx1(instance *cim.WmiInstance) (newInstance *Win32_ProductCheck, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ProductCheck{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ProductCheckEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ProductCheck, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ProductCheck{
		WmiInstance: tmp,
	}
	return
}

// SetCheck sets the value of Check for the instance
func (instance *Win32_ProductCheck) SetPropertyCheck(value CIM_Check) (err error) {
	return instance.SetProperty("Check", (value))
}

// GetCheck gets the value of Check for the instance
func (instance *Win32_ProductCheck) GetPropertyCheck() (value CIM_Check, err error) {
	retValue, err := instance.GetProperty("Check")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Check)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Check is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Check(valuetmp)

	return
}

// SetProduct sets the value of Product for the instance
func (instance *Win32_ProductCheck) SetPropertyProduct(value Win32_Product) (err error) {
	return instance.SetProperty("Product", (value))
}

// GetProduct gets the value of Product for the instance
func (instance *Win32_ProductCheck) GetPropertyProduct() (value Win32_Product, err error) {
	retValue, err := instance.GetProperty("Product")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_Product)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_Product is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_Product(valuetmp)

	return
}
