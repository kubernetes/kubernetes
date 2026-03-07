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

// CIM_ActionSequence struct
type CIM_ActionSequence struct {
	*cim.WmiInstance

	//
	Next CIM_Action

	//
	Prior CIM_Action
}

func NewCIM_ActionSequenceEx1(instance *cim.WmiInstance) (newInstance *CIM_ActionSequence, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_ActionSequence{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ActionSequenceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ActionSequence, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ActionSequence{
		WmiInstance: tmp,
	}
	return
}

// SetNext sets the value of Next for the instance
func (instance *CIM_ActionSequence) SetPropertyNext(value CIM_Action) (err error) {
	return instance.SetProperty("Next", (value))
}

// GetNext gets the value of Next for the instance
func (instance *CIM_ActionSequence) GetPropertyNext() (value CIM_Action, err error) {
	retValue, err := instance.GetProperty("Next")
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

// SetPrior sets the value of Prior for the instance
func (instance *CIM_ActionSequence) SetPropertyPrior(value CIM_Action) (err error) {
	return instance.SetProperty("Prior", (value))
}

// GetPrior gets the value of Prior for the instance
func (instance *CIM_ActionSequence) GetPropertyPrior() (value CIM_Action, err error) {
	retValue, err := instance.GetProperty("Prior")
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
