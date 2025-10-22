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

// CIM_DependencyContext struct
type CIM_DependencyContext struct {
	*cim.WmiInstance

	//
	Context CIM_Configuration

	//
	Dependency CIM_Dependency
}

func NewCIM_DependencyContextEx1(instance *cim.WmiInstance) (newInstance *CIM_DependencyContext, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_DependencyContext{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_DependencyContextEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DependencyContext, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DependencyContext{
		WmiInstance: tmp,
	}
	return
}

// SetContext sets the value of Context for the instance
func (instance *CIM_DependencyContext) SetPropertyContext(value CIM_Configuration) (err error) {
	return instance.SetProperty("Context", (value))
}

// GetContext gets the value of Context for the instance
func (instance *CIM_DependencyContext) GetPropertyContext() (value CIM_Configuration, err error) {
	retValue, err := instance.GetProperty("Context")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Configuration)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Configuration is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Configuration(valuetmp)

	return
}

// SetDependency sets the value of Dependency for the instance
func (instance *CIM_DependencyContext) SetPropertyDependency(value CIM_Dependency) (err error) {
	return instance.SetProperty("Dependency", (value))
}

// GetDependency gets the value of Dependency for the instance
func (instance *CIM_DependencyContext) GetPropertyDependency() (value CIM_Dependency, err error) {
	retValue, err := instance.GetProperty("Dependency")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Dependency)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Dependency is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Dependency(valuetmp)

	return
}
