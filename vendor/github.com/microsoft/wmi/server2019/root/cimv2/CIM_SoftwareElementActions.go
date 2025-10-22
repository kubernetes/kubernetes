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

// CIM_SoftwareElementActions struct
type CIM_SoftwareElementActions struct {
	*cim.WmiInstance

	//
	Action CIM_Action

	//
	Element CIM_SoftwareElement
}

func NewCIM_SoftwareElementActionsEx1(instance *cim.WmiInstance) (newInstance *CIM_SoftwareElementActions, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_SoftwareElementActions{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_SoftwareElementActionsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SoftwareElementActions, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SoftwareElementActions{
		WmiInstance: tmp,
	}
	return
}

// SetAction sets the value of Action for the instance
func (instance *CIM_SoftwareElementActions) SetPropertyAction(value CIM_Action) (err error) {
	return instance.SetProperty("Action", (value))
}

// GetAction gets the value of Action for the instance
func (instance *CIM_SoftwareElementActions) GetPropertyAction() (value CIM_Action, err error) {
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
func (instance *CIM_SoftwareElementActions) SetPropertyElement(value CIM_SoftwareElement) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *CIM_SoftwareElementActions) GetPropertyElement() (value CIM_SoftwareElement, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_SoftwareElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_SoftwareElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_SoftwareElement(valuetmp)

	return
}
