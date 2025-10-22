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

// MSFT_WmiError struct
type MSFT_WmiError struct {
	*CIM_Error

	// Error Category.
	error_Category uint16

	// Error code.
	error_Code uint32

	// Error Type.
	error_Type string

	// Windows error message.
	error_WindowsErrorMessage string
}

func NewMSFT_WmiErrorEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiError, err error) {
	tmp, err := NewCIM_ErrorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiError{
		CIM_Error: tmp,
	}
	return
}

func NewMSFT_WmiErrorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiError, err error) {
	tmp, err := NewCIM_ErrorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiError{
		CIM_Error: tmp,
	}
	return
}

// Seterror_Category sets the value of error_Category for the instance
func (instance *MSFT_WmiError) SetPropertyerror_Category(value uint16) (err error) {
	return instance.SetProperty("error_Category", (value))
}

// Geterror_Category gets the value of error_Category for the instance
func (instance *MSFT_WmiError) GetPropertyerror_Category() (value uint16, err error) {
	retValue, err := instance.GetProperty("error_Category")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// Seterror_Code sets the value of error_Code for the instance
func (instance *MSFT_WmiError) SetPropertyerror_Code(value uint32) (err error) {
	return instance.SetProperty("error_Code", (value))
}

// Geterror_Code gets the value of error_Code for the instance
func (instance *MSFT_WmiError) GetPropertyerror_Code() (value uint32, err error) {
	retValue, err := instance.GetProperty("error_Code")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// Seterror_Type sets the value of error_Type for the instance
func (instance *MSFT_WmiError) SetPropertyerror_Type(value string) (err error) {
	return instance.SetProperty("error_Type", (value))
}

// Geterror_Type gets the value of error_Type for the instance
func (instance *MSFT_WmiError) GetPropertyerror_Type() (value string, err error) {
	retValue, err := instance.GetProperty("error_Type")
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

// Seterror_WindowsErrorMessage sets the value of error_WindowsErrorMessage for the instance
func (instance *MSFT_WmiError) SetPropertyerror_WindowsErrorMessage(value string) (err error) {
	return instance.SetProperty("error_WindowsErrorMessage", (value))
}

// Geterror_WindowsErrorMessage gets the value of error_WindowsErrorMessage for the instance
func (instance *MSFT_WmiError) GetPropertyerror_WindowsErrorMessage() (value string, err error) {
	retValue, err := instance.GetProperty("error_WindowsErrorMessage")
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
