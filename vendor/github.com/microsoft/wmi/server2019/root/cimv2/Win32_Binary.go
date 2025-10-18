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

// Win32_Binary struct
type Win32_Binary struct {
	*Win32_MSIResource

	//
	Data string

	//
	Name string

	//
	ProductCode string
}

func NewWin32_BinaryEx1(instance *cim.WmiInstance) (newInstance *Win32_Binary, err error) {
	tmp, err := NewWin32_MSIResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Binary{
		Win32_MSIResource: tmp,
	}
	return
}

func NewWin32_BinaryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Binary, err error) {
	tmp, err := NewWin32_MSIResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Binary{
		Win32_MSIResource: tmp,
	}
	return
}

// SetData sets the value of Data for the instance
func (instance *Win32_Binary) SetPropertyData(value string) (err error) {
	return instance.SetProperty("Data", (value))
}

// GetData gets the value of Data for the instance
func (instance *Win32_Binary) GetPropertyData() (value string, err error) {
	retValue, err := instance.GetProperty("Data")
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

// SetName sets the value of Name for the instance
func (instance *Win32_Binary) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_Binary) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetProductCode sets the value of ProductCode for the instance
func (instance *Win32_Binary) SetPropertyProductCode(value string) (err error) {
	return instance.SetProperty("ProductCode", (value))
}

// GetProductCode gets the value of ProductCode for the instance
func (instance *Win32_Binary) GetPropertyProductCode() (value string, err error) {
	retValue, err := instance.GetProperty("ProductCode")
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
