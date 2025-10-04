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

// Win32_RegistryAction struct
type Win32_RegistryAction struct {
	*CIM_Action

	//
	EntryName string

	//
	EntryValue string

	//
	key string

	//
	Registry string

	//
	Root int16
}

func NewWin32_RegistryActionEx1(instance *cim.WmiInstance) (newInstance *Win32_RegistryAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_RegistryAction{
		CIM_Action: tmp,
	}
	return
}

func NewWin32_RegistryActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_RegistryAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_RegistryAction{
		CIM_Action: tmp,
	}
	return
}

// SetEntryName sets the value of EntryName for the instance
func (instance *Win32_RegistryAction) SetPropertyEntryName(value string) (err error) {
	return instance.SetProperty("EntryName", (value))
}

// GetEntryName gets the value of EntryName for the instance
func (instance *Win32_RegistryAction) GetPropertyEntryName() (value string, err error) {
	retValue, err := instance.GetProperty("EntryName")
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

// SetEntryValue sets the value of EntryValue for the instance
func (instance *Win32_RegistryAction) SetPropertyEntryValue(value string) (err error) {
	return instance.SetProperty("EntryValue", (value))
}

// GetEntryValue gets the value of EntryValue for the instance
func (instance *Win32_RegistryAction) GetPropertyEntryValue() (value string, err error) {
	retValue, err := instance.GetProperty("EntryValue")
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

// Setkey sets the value of key for the instance
func (instance *Win32_RegistryAction) SetPropertykey(value string) (err error) {
	return instance.SetProperty("key", (value))
}

// Getkey gets the value of key for the instance
func (instance *Win32_RegistryAction) GetPropertykey() (value string, err error) {
	retValue, err := instance.GetProperty("key")
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

// SetRegistry sets the value of Registry for the instance
func (instance *Win32_RegistryAction) SetPropertyRegistry(value string) (err error) {
	return instance.SetProperty("Registry", (value))
}

// GetRegistry gets the value of Registry for the instance
func (instance *Win32_RegistryAction) GetPropertyRegistry() (value string, err error) {
	retValue, err := instance.GetProperty("Registry")
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

// SetRoot sets the value of Root for the instance
func (instance *Win32_RegistryAction) SetPropertyRoot(value int16) (err error) {
	return instance.SetProperty("Root", (value))
}

// GetRoot gets the value of Root for the instance
func (instance *Win32_RegistryAction) GetPropertyRoot() (value int16, err error) {
	retValue, err := instance.GetProperty("Root")
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
