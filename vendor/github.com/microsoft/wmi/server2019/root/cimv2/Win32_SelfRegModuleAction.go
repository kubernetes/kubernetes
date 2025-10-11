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

// Win32_SelfRegModuleAction struct
type Win32_SelfRegModuleAction struct {
	*CIM_Action

	//
	Cost uint16

	//
	File string
}

func NewWin32_SelfRegModuleActionEx1(instance *cim.WmiInstance) (newInstance *Win32_SelfRegModuleAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SelfRegModuleAction{
		CIM_Action: tmp,
	}
	return
}

func NewWin32_SelfRegModuleActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SelfRegModuleAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SelfRegModuleAction{
		CIM_Action: tmp,
	}
	return
}

// SetCost sets the value of Cost for the instance
func (instance *Win32_SelfRegModuleAction) SetPropertyCost(value uint16) (err error) {
	return instance.SetProperty("Cost", (value))
}

// GetCost gets the value of Cost for the instance
func (instance *Win32_SelfRegModuleAction) GetPropertyCost() (value uint16, err error) {
	retValue, err := instance.GetProperty("Cost")
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

// SetFile sets the value of File for the instance
func (instance *Win32_SelfRegModuleAction) SetPropertyFile(value string) (err error) {
	return instance.SetProperty("File", (value))
}

// GetFile gets the value of File for the instance
func (instance *Win32_SelfRegModuleAction) GetPropertyFile() (value string, err error) {
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
