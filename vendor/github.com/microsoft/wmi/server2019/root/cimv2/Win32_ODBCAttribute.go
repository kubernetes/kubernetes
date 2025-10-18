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

// Win32_ODBCAttribute struct
type Win32_ODBCAttribute struct {
	*CIM_Setting

	//
	Attribute string

	//
	Driver string

	//
	Value string
}

func NewWin32_ODBCAttributeEx1(instance *cim.WmiInstance) (newInstance *Win32_ODBCAttribute, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCAttribute{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_ODBCAttributeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ODBCAttribute, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCAttribute{
		CIM_Setting: tmp,
	}
	return
}

// SetAttribute sets the value of Attribute for the instance
func (instance *Win32_ODBCAttribute) SetPropertyAttribute(value string) (err error) {
	return instance.SetProperty("Attribute", (value))
}

// GetAttribute gets the value of Attribute for the instance
func (instance *Win32_ODBCAttribute) GetPropertyAttribute() (value string, err error) {
	retValue, err := instance.GetProperty("Attribute")
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

// SetDriver sets the value of Driver for the instance
func (instance *Win32_ODBCAttribute) SetPropertyDriver(value string) (err error) {
	return instance.SetProperty("Driver", (value))
}

// GetDriver gets the value of Driver for the instance
func (instance *Win32_ODBCAttribute) GetPropertyDriver() (value string, err error) {
	retValue, err := instance.GetProperty("Driver")
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
func (instance *Win32_ODBCAttribute) SetPropertyValue(value string) (err error) {
	return instance.SetProperty("Value", (value))
}

// GetValue gets the value of Value for the instance
func (instance *Win32_ODBCAttribute) GetPropertyValue() (value string, err error) {
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
