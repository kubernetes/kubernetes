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

// Win32_SoftwareElement struct
type Win32_SoftwareElement struct {
	*CIM_SoftwareElement

	//
	Attributes uint16

	//
	InstallState int16

	//
	Path string
}

func NewWin32_SoftwareElementEx1(instance *cim.WmiInstance) (newInstance *Win32_SoftwareElement, err error) {
	tmp, err := NewCIM_SoftwareElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareElement{
		CIM_SoftwareElement: tmp,
	}
	return
}

func NewWin32_SoftwareElementEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SoftwareElement, err error) {
	tmp, err := NewCIM_SoftwareElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareElement{
		CIM_SoftwareElement: tmp,
	}
	return
}

// SetAttributes sets the value of Attributes for the instance
func (instance *Win32_SoftwareElement) SetPropertyAttributes(value uint16) (err error) {
	return instance.SetProperty("Attributes", (value))
}

// GetAttributes gets the value of Attributes for the instance
func (instance *Win32_SoftwareElement) GetPropertyAttributes() (value uint16, err error) {
	retValue, err := instance.GetProperty("Attributes")
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

// SetInstallState sets the value of InstallState for the instance
func (instance *Win32_SoftwareElement) SetPropertyInstallState(value int16) (err error) {
	return instance.SetProperty("InstallState", (value))
}

// GetInstallState gets the value of InstallState for the instance
func (instance *Win32_SoftwareElement) GetPropertyInstallState() (value int16, err error) {
	retValue, err := instance.GetProperty("InstallState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = int16(valuetmp)

	return
}

// SetPath sets the value of Path for the instance
func (instance *Win32_SoftwareElement) SetPropertyPath(value string) (err error) {
	return instance.SetProperty("Path", (value))
}

// GetPath gets the value of Path for the instance
func (instance *Win32_SoftwareElement) GetPropertyPath() (value string, err error) {
	retValue, err := instance.GetProperty("Path")
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
