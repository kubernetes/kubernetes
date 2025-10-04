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

// Win32_ActionCheck struct
type Win32_ActionCheck struct {
	*cim.WmiInstance

	//
	Action CIM_Action

	//
	Check CIM_Check
}

func NewWin32_ActionCheckEx1(instance *cim.WmiInstance) (newInstance *Win32_ActionCheck, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ActionCheck{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ActionCheckEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ActionCheck, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ActionCheck{
		WmiInstance: tmp,
	}
	return
}

// SetAction sets the value of Action for the instance
func (instance *Win32_ActionCheck) SetPropertyAction(value CIM_Action) (err error) {
	return instance.SetProperty("Action", (value))
}

// GetAction gets the value of Action for the instance
func (instance *Win32_ActionCheck) GetPropertyAction() (value CIM_Action, err error) {
	retValue, err := instance.GetProperty("Action")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Action)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Action is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Action(valuetmp)

	return
}

// SetCheck sets the value of Check for the instance
func (instance *Win32_ActionCheck) SetPropertyCheck(value CIM_Check) (err error) {
	return instance.SetProperty("Check", (value))
}

// GetCheck gets the value of Check for the instance
func (instance *Win32_ActionCheck) GetPropertyCheck() (value CIM_Check, err error) {
	retValue, err := instance.GetProperty("Check")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Check)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Check is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Check(valuetmp)

	return
}
