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

// Win32_EnvironmentSpecification struct
type Win32_EnvironmentSpecification struct {
	*CIM_Check

	//
	Environment string

	//
	Value string
}

func NewWin32_EnvironmentSpecificationEx1(instance *cim.WmiInstance) (newInstance *Win32_EnvironmentSpecification, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_EnvironmentSpecification{
		CIM_Check: tmp,
	}
	return
}

func NewWin32_EnvironmentSpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_EnvironmentSpecification, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_EnvironmentSpecification{
		CIM_Check: tmp,
	}
	return
}

// SetEnvironment sets the value of Environment for the instance
func (instance *Win32_EnvironmentSpecification) SetPropertyEnvironment(value string) (err error) {
	return instance.SetProperty("Environment", (value))
}

// GetEnvironment gets the value of Environment for the instance
func (instance *Win32_EnvironmentSpecification) GetPropertyEnvironment() (value string, err error) {
	retValue, err := instance.GetProperty("Environment")
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
func (instance *Win32_EnvironmentSpecification) SetPropertyValue(value string) (err error) {
	return instance.SetProperty("Value", (value))
}

// GetValue gets the value of Value for the instance
func (instance *Win32_EnvironmentSpecification) GetPropertyValue() (value string, err error) {
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
