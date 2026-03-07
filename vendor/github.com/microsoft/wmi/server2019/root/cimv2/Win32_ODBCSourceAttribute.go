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

// Win32_ODBCSourceAttribute struct
type Win32_ODBCSourceAttribute struct {
	*CIM_Setting

	//
	Attribute string

	//
	DataSource string

	//
	Value string
}

func NewWin32_ODBCSourceAttributeEx1(instance *cim.WmiInstance) (newInstance *Win32_ODBCSourceAttribute, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCSourceAttribute{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_ODBCSourceAttributeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ODBCSourceAttribute, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCSourceAttribute{
		CIM_Setting: tmp,
	}
	return
}

// SetAttribute sets the value of Attribute for the instance
func (instance *Win32_ODBCSourceAttribute) SetPropertyAttribute(value string) (err error) {
	return instance.SetProperty("Attribute", (value))
}

// GetAttribute gets the value of Attribute for the instance
func (instance *Win32_ODBCSourceAttribute) GetPropertyAttribute() (value string, err error) {
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

// SetDataSource sets the value of DataSource for the instance
func (instance *Win32_ODBCSourceAttribute) SetPropertyDataSource(value string) (err error) {
	return instance.SetProperty("DataSource", (value))
}

// GetDataSource gets the value of DataSource for the instance
func (instance *Win32_ODBCSourceAttribute) GetPropertyDataSource() (value string, err error) {
	retValue, err := instance.GetProperty("DataSource")
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
func (instance *Win32_ODBCSourceAttribute) SetPropertyValue(value string) (err error) {
	return instance.SetProperty("Value", (value))
}

// GetValue gets the value of Value for the instance
func (instance *Win32_ODBCSourceAttribute) GetPropertyValue() (value string, err error) {
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
