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

// CIM_Dependency struct
type CIM_Dependency struct {
	*cim.WmiInstance

	//
	Antecedent CIM_ManagedSystemElement

	//
	Dependent CIM_ManagedSystemElement
}

func NewCIM_DependencyEx1(instance *cim.WmiInstance) (newInstance *CIM_Dependency, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_Dependency{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_DependencyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_Dependency, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_Dependency{
		WmiInstance: tmp,
	}
	return
}

// SetAntecedent sets the value of Antecedent for the instance
func (instance *CIM_Dependency) SetPropertyAntecedent(value CIM_ManagedSystemElement) (err error) {
	return instance.SetProperty("Antecedent", (value))
}

// GetAntecedent gets the value of Antecedent for the instance
func (instance *CIM_Dependency) GetPropertyAntecedent() (value CIM_ManagedSystemElement, err error) {
	retValue, err := instance.GetProperty("Antecedent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_ManagedSystemElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_ManagedSystemElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_ManagedSystemElement(valuetmp)

	return
}

// SetDependent sets the value of Dependent for the instance
func (instance *CIM_Dependency) SetPropertyDependent(value CIM_ManagedSystemElement) (err error) {
	return instance.SetProperty("Dependent", (value))
}

// GetDependent gets the value of Dependent for the instance
func (instance *CIM_Dependency) GetPropertyDependent() (value CIM_ManagedSystemElement, err error) {
	retValue, err := instance.GetProperty("Dependent")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_ManagedSystemElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_ManagedSystemElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_ManagedSystemElement(valuetmp)

	return
}
