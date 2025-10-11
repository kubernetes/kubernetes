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

// CIM_FRUPhysicalElements struct
type CIM_FRUPhysicalElements struct {
	*cim.WmiInstance

	//
	Component CIM_PhysicalElement

	//
	FRU CIM_FRU
}

func NewCIM_FRUPhysicalElementsEx1(instance *cim.WmiInstance) (newInstance *CIM_FRUPhysicalElements, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_FRUPhysicalElements{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_FRUPhysicalElementsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_FRUPhysicalElements, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_FRUPhysicalElements{
		WmiInstance: tmp,
	}
	return
}

// SetComponent sets the value of Component for the instance
func (instance *CIM_FRUPhysicalElements) SetPropertyComponent(value CIM_PhysicalElement) (err error) {
	return instance.SetProperty("Component", (value))
}

// GetComponent gets the value of Component for the instance
func (instance *CIM_FRUPhysicalElements) GetPropertyComponent() (value CIM_PhysicalElement, err error) {
	retValue, err := instance.GetProperty("Component")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_PhysicalElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_PhysicalElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_PhysicalElement(valuetmp)

	return
}

// SetFRU sets the value of FRU for the instance
func (instance *CIM_FRUPhysicalElements) SetPropertyFRU(value CIM_FRU) (err error) {
	return instance.SetProperty("FRU", (value))
}

// GetFRU gets the value of FRU for the instance
func (instance *CIM_FRUPhysicalElements) GetPropertyFRU() (value CIM_FRU, err error) {
	retValue, err := instance.GetProperty("FRU")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_FRU)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_FRU is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_FRU(valuetmp)

	return
}
