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

// CIM_SoftwareElementChecks struct
type CIM_SoftwareElementChecks struct {
	*cim.WmiInstance

	//
	Check CIM_Check

	//
	Element CIM_SoftwareElement

	//
	Phase uint16
}

func NewCIM_SoftwareElementChecksEx1(instance *cim.WmiInstance) (newInstance *CIM_SoftwareElementChecks, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_SoftwareElementChecks{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_SoftwareElementChecksEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SoftwareElementChecks, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SoftwareElementChecks{
		WmiInstance: tmp,
	}
	return
}

// SetCheck sets the value of Check for the instance
func (instance *CIM_SoftwareElementChecks) SetPropertyCheck(value CIM_Check) (err error) {
	return instance.SetProperty("Check", (value))
}

// GetCheck gets the value of Check for the instance
func (instance *CIM_SoftwareElementChecks) GetPropertyCheck() (value CIM_Check, err error) {
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

// SetElement sets the value of Element for the instance
func (instance *CIM_SoftwareElementChecks) SetPropertyElement(value CIM_SoftwareElement) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *CIM_SoftwareElementChecks) GetPropertyElement() (value CIM_SoftwareElement, err error) {
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

// SetPhase sets the value of Phase for the instance
func (instance *CIM_SoftwareElementChecks) SetPropertyPhase(value uint16) (err error) {
	return instance.SetProperty("Phase", (value))
}

// GetPhase gets the value of Phase for the instance
func (instance *CIM_SoftwareElementChecks) GetPropertyPhase() (value uint16, err error) {
	retValue, err := instance.GetProperty("Phase")
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
