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

// Win32_BootConfiguration struct
type Win32_BootConfiguration struct {
	*CIM_Setting

	//
	BootDirectory string

	//
	ConfigurationPath string

	//
	LastDrive string

	//
	Name string

	//
	ScratchDirectory string

	//
	TempDirectory string
}

func NewWin32_BootConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_BootConfiguration, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_BootConfiguration{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_BootConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_BootConfiguration, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_BootConfiguration{
		CIM_Setting: tmp,
	}
	return
}

// SetBootDirectory sets the value of BootDirectory for the instance
func (instance *Win32_BootConfiguration) SetPropertyBootDirectory(value string) (err error) {
	return instance.SetProperty("BootDirectory", (value))
}

// GetBootDirectory gets the value of BootDirectory for the instance
func (instance *Win32_BootConfiguration) GetPropertyBootDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("BootDirectory")
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

// SetConfigurationPath sets the value of ConfigurationPath for the instance
func (instance *Win32_BootConfiguration) SetPropertyConfigurationPath(value string) (err error) {
	return instance.SetProperty("ConfigurationPath", (value))
}

// GetConfigurationPath gets the value of ConfigurationPath for the instance
func (instance *Win32_BootConfiguration) GetPropertyConfigurationPath() (value string, err error) {
	retValue, err := instance.GetProperty("ConfigurationPath")
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

// SetLastDrive sets the value of LastDrive for the instance
func (instance *Win32_BootConfiguration) SetPropertyLastDrive(value string) (err error) {
	return instance.SetProperty("LastDrive", (value))
}

// GetLastDrive gets the value of LastDrive for the instance
func (instance *Win32_BootConfiguration) GetPropertyLastDrive() (value string, err error) {
	retValue, err := instance.GetProperty("LastDrive")
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
func (instance *Win32_BootConfiguration) SetPropertyName(value string) (err error) {
	return instance.SetProperty("Name", (value))
}

// GetName gets the value of Name for the instance
func (instance *Win32_BootConfiguration) GetPropertyName() (value string, err error) {
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

// SetScratchDirectory sets the value of ScratchDirectory for the instance
func (instance *Win32_BootConfiguration) SetPropertyScratchDirectory(value string) (err error) {
	return instance.SetProperty("ScratchDirectory", (value))
}

// GetScratchDirectory gets the value of ScratchDirectory for the instance
func (instance *Win32_BootConfiguration) GetPropertyScratchDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("ScratchDirectory")
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

// SetTempDirectory sets the value of TempDirectory for the instance
func (instance *Win32_BootConfiguration) SetPropertyTempDirectory(value string) (err error) {
	return instance.SetProperty("TempDirectory", (value))
}

// GetTempDirectory gets the value of TempDirectory for the instance
func (instance *Win32_BootConfiguration) GetPropertyTempDirectory() (value string, err error) {
	retValue, err := instance.GetProperty("TempDirectory")
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
