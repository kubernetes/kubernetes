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

// Win32_ODBCTranslatorSpecification struct
type Win32_ODBCTranslatorSpecification struct {
	*CIM_Check

	//
	File string

	//
	SetupFile string

	//
	Translator string
}

func NewWin32_ODBCTranslatorSpecificationEx1(instance *cim.WmiInstance) (newInstance *Win32_ODBCTranslatorSpecification, err error) {
	tmp, err := NewCIM_CheckEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCTranslatorSpecification{
		CIM_Check: tmp,
	}
	return
}

func NewWin32_ODBCTranslatorSpecificationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ODBCTranslatorSpecification, err error) {
	tmp, err := NewCIM_CheckEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCTranslatorSpecification{
		CIM_Check: tmp,
	}
	return
}

// SetFile sets the value of File for the instance
func (instance *Win32_ODBCTranslatorSpecification) SetPropertyFile(value string) (err error) {
	return instance.SetProperty("File", (value))
}

// GetFile gets the value of File for the instance
func (instance *Win32_ODBCTranslatorSpecification) GetPropertyFile() (value string, err error) {
	retValue, err := instance.GetProperty("File")
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

// SetSetupFile sets the value of SetupFile for the instance
func (instance *Win32_ODBCTranslatorSpecification) SetPropertySetupFile(value string) (err error) {
	return instance.SetProperty("SetupFile", (value))
}

// GetSetupFile gets the value of SetupFile for the instance
func (instance *Win32_ODBCTranslatorSpecification) GetPropertySetupFile() (value string, err error) {
	retValue, err := instance.GetProperty("SetupFile")
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

// SetTranslator sets the value of Translator for the instance
func (instance *Win32_ODBCTranslatorSpecification) SetPropertyTranslator(value string) (err error) {
	return instance.SetProperty("Translator", (value))
}

// GetTranslator gets the value of Translator for the instance
func (instance *Win32_ODBCTranslatorSpecification) GetPropertyTranslator() (value string, err error) {
	retValue, err := instance.GetProperty("Translator")
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
