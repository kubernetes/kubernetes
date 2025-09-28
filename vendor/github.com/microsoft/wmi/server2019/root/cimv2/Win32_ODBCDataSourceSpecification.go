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

// Win32_ODBCDataSourceSpecification struct
type Win32_ODBCDataSourceSpecification struct {
	*CIM_Check

	//
	DataSource string

	//
	DriverDescription string

	//
	Registration string
}

func NewWin32_ODBCDataSourceSpecificationEx1(instance *cim.WmiInstance) (newInstance *Win32_ODBCDataSourceSpecification, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCDataSourceSpecification{
		CIM_Check: tmp,
	}
	return
}

func NewWin32_ODBCDataSourceSpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ODBCDataSourceSpecification, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCDataSourceSpecification{
		CIM_Check: tmp,
	}
	return
}

// SetDataSource sets the value of DataSource for the instance
func (instance *Win32_ODBCDataSourceSpecification) SetPropertyDataSource(value string) (err error) {
	return instance.SetProperty("DataSource", (value))
}

// GetDataSource gets the value of DataSource for the instance
func (instance *Win32_ODBCDataSourceSpecification) GetPropertyDataSource() (value string, err error) {
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

// SetDriverDescription sets the value of DriverDescription for the instance
func (instance *Win32_ODBCDataSourceSpecification) SetPropertyDriverDescription(value string) (err error) {
	return instance.SetProperty("DriverDescription", (value))
}

// GetDriverDescription gets the value of DriverDescription for the instance
func (instance *Win32_ODBCDataSourceSpecification) GetPropertyDriverDescription() (value string, err error) {
	retValue, err := instance.GetProperty("DriverDescription")
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

// SetRegistration sets the value of Registration for the instance
func (instance *Win32_ODBCDataSourceSpecification) SetPropertyRegistration(value string) (err error) {
	return instance.SetProperty("Registration", (value))
}

// GetRegistration gets the value of Registration for the instance
func (instance *Win32_ODBCDataSourceSpecification) GetPropertyRegistration() (value string, err error) {
	retValue, err := instance.GetProperty("Registration")
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
