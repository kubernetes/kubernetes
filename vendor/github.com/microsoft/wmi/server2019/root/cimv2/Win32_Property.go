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

// Win32_Property struct
type Win32_Property struct {
	*Win32_MSIResource

	//
	ProductCode string

	//
	Property string

	//
	Value string
}

func NewWin32_PropertyEx1(instance *cim.WmiInstance) (newInstance *Win32_Property, err error) {
	tmp, err := NewWin32_MSIResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Property{
		Win32_MSIResource: tmp,
	}
	return
}

func NewWin32_PropertyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Property, err error) {
	tmp, err := NewWin32_MSIResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Property{
		Win32_MSIResource: tmp,
	}
	return
}

// SetProductCode sets the value of ProductCode for the instance
func (instance *Win32_Property) SetPropertyProductCode(value string) (err error) {
	return instance.SetProperty("ProductCode", (value))
}

// GetProductCode gets the value of ProductCode for the instance
func (instance *Win32_Property) GetPropertyProductCode() (value string, err error) {
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

// SetProperty sets the value of Property for the instance
func (instance *Win32_Property) SetPropertyProperty(value string) (err error) {
	return instance.SetProperty("Property", (value))
}

// GetProperty gets the value of Property for the instance
func (instance *Win32_Property) GetPropertyProperty() (value string, err error) {
	retValue, err := instance.GetProperty("Property")
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

// SetValue sets the value of Value for the instance
func (instance *Win32_Property) SetPropertyValue(value string) (err error) {
	return instance.SetProperty("Value", (value))
}

// GetValue gets the value of Value for the instance
func (instance *Win32_Property) GetPropertyValue() (value string, err error) {
	retValue, err := instance.GetProperty("Value")
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
