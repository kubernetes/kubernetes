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

// Win32_QuickFixEngineering struct
type Win32_QuickFixEngineering struct {
	*CIM_LogicalElement

	//
	CSName string

	//
	FixComments string

	//
	HotFixID string

	//
	InstalledBy string

	//
	InstalledOn string

	//
	ServicePackInEffect string
}

func NewWin32_QuickFixEngineeringEx1(instance *cim.WmiInstance) (newInstance *Win32_QuickFixEngineering, err error) {
	tmp, err := NewCIM_LogicalElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_QuickFixEngineering{
		CIM_LogicalElement: tmp,
	}
	return
}

func NewWin32_QuickFixEngineeringEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_QuickFixEngineering, err error) {
	tmp, err := NewCIM_LogicalElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_QuickFixEngineering{
		CIM_LogicalElement: tmp,
	}
	return
}

// SetCSName sets the value of CSName for the instance
func (instance *Win32_QuickFixEngineering) SetPropertyCSName(value string) (err error) {
	return instance.SetProperty("CSName", (value))
}

// GetCSName gets the value of CSName for the instance
func (instance *Win32_QuickFixEngineering) GetPropertyCSName() (value string, err error) {
	retValue, err := instance.GetProperty("CSName")
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

// SetFixComments sets the value of FixComments for the instance
func (instance *Win32_QuickFixEngineering) SetPropertyFixComments(value string) (err error) {
	return instance.SetProperty("FixComments", (value))
}

// GetFixComments gets the value of FixComments for the instance
func (instance *Win32_QuickFixEngineering) GetPropertyFixComments() (value string, err error) {
	retValue, err := instance.GetProperty("FixComments")
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

// SetHotFixID sets the value of HotFixID for the instance
func (instance *Win32_QuickFixEngineering) SetPropertyHotFixID(value string) (err error) {
	return instance.SetProperty("HotFixID", (value))
}

// GetHotFixID gets the value of HotFixID for the instance
func (instance *Win32_QuickFixEngineering) GetPropertyHotFixID() (value string, err error) {
	retValue, err := instance.GetProperty("HotFixID")
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

// SetInstalledBy sets the value of InstalledBy for the instance
func (instance *Win32_QuickFixEngineering) SetPropertyInstalledBy(value string) (err error) {
	return instance.SetProperty("InstalledBy", (value))
}

// GetInstalledBy gets the value of InstalledBy for the instance
func (instance *Win32_QuickFixEngineering) GetPropertyInstalledBy() (value string, err error) {
	retValue, err := instance.GetProperty("InstalledBy")
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

// SetInstalledOn sets the value of InstalledOn for the instance
func (instance *Win32_QuickFixEngineering) SetPropertyInstalledOn(value string) (err error) {
	return instance.SetProperty("InstalledOn", (value))
}

// GetInstalledOn gets the value of InstalledOn for the instance
func (instance *Win32_QuickFixEngineering) GetPropertyInstalledOn() (value string, err error) {
	retValue, err := instance.GetProperty("InstalledOn")
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

// SetServicePackInEffect sets the value of ServicePackInEffect for the instance
func (instance *Win32_QuickFixEngineering) SetPropertyServicePackInEffect(value string) (err error) {
	return instance.SetProperty("ServicePackInEffect", (value))
}

// GetServicePackInEffect gets the value of ServicePackInEffect for the instance
func (instance *Win32_QuickFixEngineering) GetPropertyServicePackInEffect() (value string, err error) {
	retValue, err := instance.GetProperty("ServicePackInEffect")
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
