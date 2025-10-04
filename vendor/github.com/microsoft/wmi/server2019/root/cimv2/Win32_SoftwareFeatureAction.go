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

// Win32_SoftwareFeatureAction struct
type Win32_SoftwareFeatureAction struct {
	*cim.WmiInstance

	//
	Action CIM_Action

	//
	Element Win32_SoftwareFeature
}

func NewWin32_SoftwareFeatureActionEx1(instance *cim.WmiInstance) (newInstance *Win32_SoftwareFeatureAction, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareFeatureAction{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_SoftwareFeatureActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SoftwareFeatureAction, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareFeatureAction{
		WmiInstance: tmp,
	}
	return
}

// SetAction sets the value of Action for the instance
func (instance *Win32_SoftwareFeatureAction) SetPropertyAction(value CIM_Action) (err error) {
	return instance.SetProperty("Action", (value))
}

// GetAction gets the value of Action for the instance
func (instance *Win32_SoftwareFeatureAction) GetPropertyAction() (value CIM_Action, err error) {
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

// SetElement sets the value of Element for the instance
func (instance *Win32_SoftwareFeatureAction) SetPropertyElement(value Win32_SoftwareFeature) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *Win32_SoftwareFeatureAction) GetPropertyElement() (value Win32_SoftwareFeature, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_SoftwareFeature)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_SoftwareFeature is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_SoftwareFeature(valuetmp)

	return
}
