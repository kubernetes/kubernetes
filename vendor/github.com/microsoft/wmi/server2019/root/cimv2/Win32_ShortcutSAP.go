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

// Win32_ShortcutSAP struct
type Win32_ShortcutSAP struct {
	*cim.WmiInstance

	//
	Action Win32_ShortcutAction

	//
	Element Win32_CommandLineAccess
}

func NewWin32_ShortcutSAPEx1(instance *cim.WmiInstance) (newInstance *Win32_ShortcutSAP, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ShortcutSAP{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ShortcutSAPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ShortcutSAP, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ShortcutSAP{
		WmiInstance: tmp,
	}
	return
}

// SetAction sets the value of Action for the instance
func (instance *Win32_ShortcutSAP) SetPropertyAction(value Win32_ShortcutAction) (err error) {
	return instance.SetProperty("Action", (value))
}

// GetAction gets the value of Action for the instance
func (instance *Win32_ShortcutSAP) GetPropertyAction() (value Win32_ShortcutAction, err error) {
	retValue, err := instance.GetProperty("Action")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_ShortcutAction)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_ShortcutAction is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_ShortcutAction(valuetmp)

	return
}

// SetElement sets the value of Element for the instance
func (instance *Win32_ShortcutSAP) SetPropertyElement(value Win32_CommandLineAccess) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *Win32_ShortcutSAP) GetPropertyElement() (value Win32_CommandLineAccess, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_CommandLineAccess)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_CommandLineAccess is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_CommandLineAccess(valuetmp)

	return
}
