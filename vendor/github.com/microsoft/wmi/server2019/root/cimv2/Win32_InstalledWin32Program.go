// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_InstalledWin32Program struct
type Win32_InstalledWin32Program struct {
	*cim.WmiInstance

	//
	Language string

	//
	MsiPackageCode string

	//
	MsiProductCode string

	//
	Name string

	//
	ProgramId string

	//
	Vendor string

	//
	Version string
}

func NewWin32_InstalledWin32ProgramEx1(instance *cim.WmiInstance) (newInstance *Win32_InstalledWin32Program, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_InstalledWin32Program{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_InstalledWin32ProgramEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_InstalledWin32Program, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_InstalledWin32Program{
		WmiInstance: tmp,
	}
	return
}

// SetLanguage sets the value of Language for the instance
func (instance *Win32_InstalledWin32Program) SetPropertyLanguage(value string) (err error) {
	return instance.SetProperty("Language", (value))
}

// GetLanguage gets the value of Language for the instance
func (instance *Win32_InstalledWin32Program) GetPropertyLanguage() (value string, err error) {
	retValue, err := instance.GetProperty("Language")
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

// SetMsiPackageCode sets the value of MsiPackageCode for the instance
func (instance *Win32_InstalledWin32Program) SetPropertyMsiPackageCode(value string) (err error) {
	return instance.SetProperty("MsiPackageCode", (value))
}

// GetMsiPackageCode gets the value of MsiPackageCode for the instance
func (instance *Win32_InstalledWin32Program) GetPropertyMsiPackageCode() (value string, err error) {
	retValue, err := instance.GetProperty("MsiPackageCode")
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

// SetMsiProductCode sets the value of MsiProductCode for the instance
func (instance *Win32_InstalledWin32Program) SetPropertyMsiProductCode(value string) (err error) {
	return instance.SetProperty("MsiProductCode", (value))
}

// GetMsiProductCode gets the value of MsiProductCode for the instance
func (instance *Win32_InstalledWin32Program) GetPropertyMsiProductCode() (value string, err error) {
	retValue, err := instance.GetProperty("MsiProductCode")
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

// SetName sets the value of Name for the instance
func (instance *Win32_InstalledWin32Program) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_InstalledWin32Program) GetPropertyName() (value string, err error) {
	retValue, err := instance.GetProperty("Name")
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

// SetProgramId sets the value of ProgramId for the instance
func (instance *Win32_InstalledWin32Program) SetPropertyProgramId(value string) (err error) {
	return instance.SetProperty("ProgramId", (value))
}

// GetProgramId gets the value of ProgramId for the instance
func (instance *Win32_InstalledWin32Program) GetPropertyProgramId() (value string, err error) {
	retValue, err := instance.GetProperty("ProgramId")
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

// SetVendor sets the value of Vendor for the instance
func (instance *Win32_InstalledWin32Program) SetPropertyVendor(value string) (err error) {
	return instance.SetProperty("Vendor", (value))
}

// GetVendor gets the value of Vendor for the instance
func (instance *Win32_InstalledWin32Program) GetPropertyVendor() (value string, err error) {
	retValue, err := instance.GetProperty("Vendor")
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

// SetVersion sets the value of Version for the instance
func (instance *Win32_InstalledWin32Program) SetPropertyVersion(value string) (err error) {
	return instance.SetProperty("Version", (value))
}

// GetVersion gets the value of Version for the instance
func (instance *Win32_InstalledWin32Program) GetPropertyVersion() (value string, err error) {
	retValue, err := instance.GetProperty("Version")
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
