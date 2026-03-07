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

// Win32_Environment struct
type Win32_Environment struct {
	*CIM_SystemResource

	//
	SystemVariable bool

	//
	UserName string

	//
	VariableValue string
}

func NewWin32_EnvironmentEx1(instance *cim.WmiInstance) (newInstance *Win32_Environment, err error) {
	tmp, err := NewCIM_SystemResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Environment{
		CIM_SystemResource: tmp,
	}
	return
}

func NewWin32_EnvironmentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Environment, err error) {
	tmp, err := NewCIM_SystemResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Environment{
		CIM_SystemResource: tmp,
	}
	return
}

// SetSystemVariable sets the value of SystemVariable for the instance
func (instance *Win32_Environment) SetPropertySystemVariable(value bool) (err error) {
	return instance.SetProperty("SystemVariable", (value))
}

// GetSystemVariable gets the value of SystemVariable for the instance
func (instance *Win32_Environment) GetPropertySystemVariable() (value bool, err error) {
	retValue, err := instance.GetProperty("SystemVariable")
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

// SetUserName sets the value of UserName for the instance
func (instance *Win32_Environment) SetPropertyUserName(value string) (err error) {
	return instance.SetProperty("UserName", (value))
}

// GetUserName gets the value of UserName for the instance
func (instance *Win32_Environment) GetPropertyUserName() (value string, err error) {
	retValue, err := instance.GetProperty("UserName")
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

// SetVariableValue sets the value of VariableValue for the instance
func (instance *Win32_Environment) SetPropertyVariableValue(value string) (err error) {
	return instance.SetProperty("VariableValue", (value))
}

// GetVariableValue gets the value of VariableValue for the instance
func (instance *Win32_Environment) GetPropertyVariableValue() (value string, err error) {
	retValue, err := instance.GetProperty("VariableValue")
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
