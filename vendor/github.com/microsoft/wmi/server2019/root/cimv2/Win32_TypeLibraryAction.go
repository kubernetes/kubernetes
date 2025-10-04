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

// Win32_TypeLibraryAction struct
type Win32_TypeLibraryAction struct {
	*CIM_Action

	//
	Cost uint32

	//
	Language uint16

	//
	LibID string
}

func NewWin32_TypeLibraryActionEx1(instance *cim.WmiInstance) (newInstance *Win32_TypeLibraryAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_TypeLibraryAction{
		CIM_Action: tmp,
	}
	return
}

func NewWin32_TypeLibraryActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_TypeLibraryAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_TypeLibraryAction{
		CIM_Action: tmp,
	}
	return
}

// SetCost sets the value of Cost for the instance
func (instance *Win32_TypeLibraryAction) SetPropertyCost(value uint32) (err error) {
	return instance.SetProperty("Cost", (value))
}

// GetCost gets the value of Cost for the instance
func (instance *Win32_TypeLibraryAction) GetPropertyCost() (value uint32, err error) {
	retValue, err := instance.GetProperty("Cost")
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

// SetLanguage sets the value of Language for the instance
func (instance *Win32_TypeLibraryAction) SetPropertyLanguage(value uint16) (err error) {
	return instance.SetProperty("Language", (value))
}

// GetLanguage gets the value of Language for the instance
func (instance *Win32_TypeLibraryAction) GetPropertyLanguage() (value uint16, err error) {
	retValue, err := instance.GetProperty("Language")
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

// SetLibID sets the value of LibID for the instance
func (instance *Win32_TypeLibraryAction) SetPropertyLibID(value string) (err error) {
	return instance.SetProperty("LibID", (value))
}

// GetLibID gets the value of LibID for the instance
func (instance *Win32_TypeLibraryAction) GetPropertyLibID() (value string, err error) {
	retValue, err := instance.GetProperty("LibID")
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
