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

// CIM_InstalledSoftwareElement struct
type CIM_InstalledSoftwareElement struct {
	*cim.WmiInstance

	//
	Software CIM_SoftwareElement

	//
	System CIM_ComputerSystem
}

func NewCIM_InstalledSoftwareElementEx1(instance *cim.WmiInstance) (newInstance *CIM_InstalledSoftwareElement, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_InstalledSoftwareElement{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_InstalledSoftwareElementEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_InstalledSoftwareElement, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_InstalledSoftwareElement{
		WmiInstance: tmp,
	}
	return
}

// SetSoftware sets the value of Software for the instance
func (instance *CIM_InstalledSoftwareElement) SetPropertySoftware(value CIM_SoftwareElement) (err error) {
	return instance.SetProperty("Software", (value))
}

// GetSoftware gets the value of Software for the instance
func (instance *CIM_InstalledSoftwareElement) GetPropertySoftware() (value CIM_SoftwareElement, err error) {
	retValue, err := instance.GetProperty("Software")
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

// SetSystem sets the value of System for the instance
func (instance *CIM_InstalledSoftwareElement) SetPropertySystem(value CIM_ComputerSystem) (err error) {
	return instance.SetProperty("System", (value))
}

// GetSystem gets the value of System for the instance
func (instance *CIM_InstalledSoftwareElement) GetPropertySystem() (value CIM_ComputerSystem, err error) {
	retValue, err := instance.GetProperty("System")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_ComputerSystem)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_ComputerSystem is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_ComputerSystem(valuetmp)

	return
}
