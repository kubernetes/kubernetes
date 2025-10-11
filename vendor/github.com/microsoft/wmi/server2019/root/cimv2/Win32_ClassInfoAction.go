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

// Win32_ClassInfoAction struct
type Win32_ClassInfoAction struct {
	*CIM_Action

	//
	AppID string

	//
	Argument string

	//
	CLSID string

	//
	Context string

	//
	DefInprocHandler string

	//
	FileTypeMask string

	//
	Insertable uint16

	//
	ProgID string

	//
	RemoteName string

	//
	VIProgID string
}

func NewWin32_ClassInfoActionEx1(instance *cim.WmiInstance) (newInstance *Win32_ClassInfoAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassInfoAction{
		CIM_Action: tmp,
	}
	return
}

func NewWin32_ClassInfoActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ClassInfoAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassInfoAction{
		CIM_Action: tmp,
	}
	return
}

// SetAppID sets the value of AppID for the instance
func (instance *Win32_ClassInfoAction) SetPropertyAppID(value string) (err error) {
	return instance.SetProperty("AppID", (value))
}

// GetAppID gets the value of AppID for the instance
func (instance *Win32_ClassInfoAction) GetPropertyAppID() (value string, err error) {
	retValue, err := instance.GetProperty("AppID")
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

// SetArgument sets the value of Argument for the instance
func (instance *Win32_ClassInfoAction) SetPropertyArgument(value string) (err error) {
	return instance.SetProperty("Argument", (value))
}

// GetArgument gets the value of Argument for the instance
func (instance *Win32_ClassInfoAction) GetPropertyArgument() (value string, err error) {
	retValue, err := instance.GetProperty("Argument")
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

// SetCLSID sets the value of CLSID for the instance
func (instance *Win32_ClassInfoAction) SetPropertyCLSID(value string) (err error) {
	return instance.SetProperty("CLSID", (value))
}

// GetCLSID gets the value of CLSID for the instance
func (instance *Win32_ClassInfoAction) GetPropertyCLSID() (value string, err error) {
	retValue, err := instance.GetProperty("CLSID")
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

// SetContext sets the value of Context for the instance
func (instance *Win32_ClassInfoAction) SetPropertyContext(value string) (err error) {
	return instance.SetProperty("Context", (value))
}

// GetContext gets the value of Context for the instance
func (instance *Win32_ClassInfoAction) GetPropertyContext() (value string, err error) {
	retValue, err := instance.GetProperty("Context")
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

// SetDefInprocHandler sets the value of DefInprocHandler for the instance
func (instance *Win32_ClassInfoAction) SetPropertyDefInprocHandler(value string) (err error) {
	return instance.SetProperty("DefInprocHandler", (value))
}

// GetDefInprocHandler gets the value of DefInprocHandler for the instance
func (instance *Win32_ClassInfoAction) GetPropertyDefInprocHandler() (value string, err error) {
	retValue, err := instance.GetProperty("DefInprocHandler")
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

// SetFileTypeMask sets the value of FileTypeMask for the instance
func (instance *Win32_ClassInfoAction) SetPropertyFileTypeMask(value string) (err error) {
	return instance.SetProperty("FileTypeMask", (value))
}

// GetFileTypeMask gets the value of FileTypeMask for the instance
func (instance *Win32_ClassInfoAction) GetPropertyFileTypeMask() (value string, err error) {
	retValue, err := instance.GetProperty("FileTypeMask")
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

// SetInsertable sets the value of Insertable for the instance
func (instance *Win32_ClassInfoAction) SetPropertyInsertable(value uint16) (err error) {
	return instance.SetProperty("Insertable", (value))
}

// GetInsertable gets the value of Insertable for the instance
func (instance *Win32_ClassInfoAction) GetPropertyInsertable() (value uint16, err error) {
	retValue, err := instance.GetProperty("Insertable")
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

// SetProgID sets the value of ProgID for the instance
func (instance *Win32_ClassInfoAction) SetPropertyProgID(value string) (err error) {
	return instance.SetProperty("ProgID", (value))
}

// GetProgID gets the value of ProgID for the instance
func (instance *Win32_ClassInfoAction) GetPropertyProgID() (value string, err error) {
	retValue, err := instance.GetProperty("ProgID")
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

// SetRemoteName sets the value of RemoteName for the instance
func (instance *Win32_ClassInfoAction) SetPropertyRemoteName(value string) (err error) {
	return instance.SetProperty("RemoteName", (value))
}

// GetRemoteName gets the value of RemoteName for the instance
func (instance *Win32_ClassInfoAction) GetPropertyRemoteName() (value string, err error) {
	retValue, err := instance.GetProperty("RemoteName")
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

// SetVIProgID sets the value of VIProgID for the instance
func (instance *Win32_ClassInfoAction) SetPropertyVIProgID(value string) (err error) {
	return instance.SetProperty("VIProgID", (value))
}

// GetVIProgID gets the value of VIProgID for the instance
func (instance *Win32_ClassInfoAction) GetPropertyVIProgID() (value string, err error) {
	retValue, err := instance.GetProperty("VIProgID")
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
