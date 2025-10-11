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

// Win32_ShortcutAction struct
type Win32_ShortcutAction struct {
	*CIM_Action

	//
	Arguments string

	//
	HotKey uint16

	//
	IconIndex string

	//
	Shortcut string

	//
	ShowCmd uint16

	//
	Target string

	//
	WkDir string
}

func NewWin32_ShortcutActionEx1(instance *cim.WmiInstance) (newInstance *Win32_ShortcutAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ShortcutAction{
		CIM_Action: tmp,
	}
	return
}

func NewWin32_ShortcutActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ShortcutAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ShortcutAction{
		CIM_Action: tmp,
	}
	return
}

// SetArguments sets the value of Arguments for the instance
func (instance *Win32_ShortcutAction) SetPropertyArguments(value string) (err error) {
	return instance.SetProperty("Arguments", (value))
}

// GetArguments gets the value of Arguments for the instance
func (instance *Win32_ShortcutAction) GetPropertyArguments() (value string, err error) {
	retValue, err := instance.GetProperty("Arguments")
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

// SetHotKey sets the value of HotKey for the instance
func (instance *Win32_ShortcutAction) SetPropertyHotKey(value uint16) (err error) {
	return instance.SetProperty("HotKey", (value))
}

// GetHotKey gets the value of HotKey for the instance
func (instance *Win32_ShortcutAction) GetPropertyHotKey() (value uint16, err error) {
	retValue, err := instance.GetProperty("HotKey")
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

// SetIconIndex sets the value of IconIndex for the instance
func (instance *Win32_ShortcutAction) SetPropertyIconIndex(value string) (err error) {
	return instance.SetProperty("IconIndex", (value))
}

// GetIconIndex gets the value of IconIndex for the instance
func (instance *Win32_ShortcutAction) GetPropertyIconIndex() (value string, err error) {
	retValue, err := instance.GetProperty("IconIndex")
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

// SetShortcut sets the value of Shortcut for the instance
func (instance *Win32_ShortcutAction) SetPropertyShortcut(value string) (err error) {
	return instance.SetProperty("Shortcut", (value))
}

// GetShortcut gets the value of Shortcut for the instance
func (instance *Win32_ShortcutAction) GetPropertyShortcut() (value string, err error) {
	retValue, err := instance.GetProperty("Shortcut")
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

// SetShowCmd sets the value of ShowCmd for the instance
func (instance *Win32_ShortcutAction) SetPropertyShowCmd(value uint16) (err error) {
	return instance.SetProperty("ShowCmd", (value))
}

// GetShowCmd gets the value of ShowCmd for the instance
func (instance *Win32_ShortcutAction) GetPropertyShowCmd() (value uint16, err error) {
	retValue, err := instance.GetProperty("ShowCmd")
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

// SetTarget sets the value of Target for the instance
func (instance *Win32_ShortcutAction) SetPropertyTarget(value string) (err error) {
	return instance.SetProperty("Target", (value))
}

// GetTarget gets the value of Target for the instance
func (instance *Win32_ShortcutAction) GetPropertyTarget() (value string, err error) {
	retValue, err := instance.GetProperty("Target")
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

// SetWkDir sets the value of WkDir for the instance
func (instance *Win32_ShortcutAction) SetPropertyWkDir(value string) (err error) {
	return instance.SetProperty("WkDir", (value))
}

// GetWkDir gets the value of WkDir for the instance
func (instance *Win32_ShortcutAction) GetPropertyWkDir() (value string, err error) {
	retValue, err := instance.GetProperty("WkDir")
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
