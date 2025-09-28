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

// CIM_PhysicalElementLocation struct
type CIM_PhysicalElementLocation struct {
	*cim.WmiInstance

	//
	Element CIM_PhysicalElement

	//
	PhysicalLocation CIM_Location
}

func NewCIM_PhysicalElementLocationEx1(instance *cim.WmiInstance) (newInstance *CIM_PhysicalElementLocation, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalElementLocation{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_PhysicalElementLocationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PhysicalElementLocation, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PhysicalElementLocation{
		WmiInstance: tmp,
	}
	return
}

// SetElement sets the value of Element for the instance
func (instance *CIM_PhysicalElementLocation) SetPropertyElement(value CIM_PhysicalElement) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *CIM_PhysicalElementLocation) GetPropertyElement() (value CIM_PhysicalElement, err error) {
	retValue, err := instance.GetProperty("Element")
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

// SetPhysicalLocation sets the value of PhysicalLocation for the instance
func (instance *CIM_PhysicalElementLocation) SetPropertyPhysicalLocation(value CIM_Location) (err error) {
	return instance.SetProperty("PhysicalLocation", (value))
}

// GetPhysicalLocation gets the value of PhysicalLocation for the instance
func (instance *CIM_PhysicalElementLocation) GetPropertyPhysicalLocation() (value CIM_Location, err error) {
	retValue, err := instance.GetProperty("PhysicalLocation")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Location)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Location is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Location(valuetmp)

	return
}
