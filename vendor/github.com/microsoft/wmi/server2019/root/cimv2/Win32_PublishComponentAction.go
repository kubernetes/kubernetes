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

// Win32_PublishComponentAction struct
type Win32_PublishComponentAction struct {
	*CIM_Action

	//
	AppData string

	//
	ComponentID string

	//
	Qual string
}

func NewWin32_PublishComponentActionEx1(instance *cim.WmiInstance) (newInstance *Win32_PublishComponentAction, err error) {
	tmp, err := NewCIM_ActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PublishComponentAction{
		CIM_Action: tmp,
	}
	return
}

func NewWin32_PublishComponentActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PublishComponentAction, err error) {
	tmp, err := NewCIM_ActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PublishComponentAction{
		CIM_Action: tmp,
	}
	return
}

// SetAppData sets the value of AppData for the instance
func (instance *Win32_PublishComponentAction) SetPropertyAppData(value string) (err error) {
	return instance.SetProperty("AppData", (value))
}

// GetAppData gets the value of AppData for the instance
func (instance *Win32_PublishComponentAction) GetPropertyAppData() (value string, err error) {
	retValue, err := instance.GetProperty("AppData")
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

// SetComponentID sets the value of ComponentID for the instance
func (instance *Win32_PublishComponentAction) SetPropertyComponentID(value string) (err error) {
	return instance.SetProperty("ComponentID", (value))
}

// GetComponentID gets the value of ComponentID for the instance
func (instance *Win32_PublishComponentAction) GetPropertyComponentID() (value string, err error) {
	retValue, err := instance.GetProperty("ComponentID")
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

// SetQual sets the value of Qual for the instance
func (instance *Win32_PublishComponentAction) SetPropertyQual(value string) (err error) {
	return instance.SetProperty("Qual", (value))
}

// GetQual gets the value of Qual for the instance
func (instance *Win32_PublishComponentAction) GetPropertyQual() (value string, err error) {
	retValue, err := instance.GetProperty("Qual")
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
