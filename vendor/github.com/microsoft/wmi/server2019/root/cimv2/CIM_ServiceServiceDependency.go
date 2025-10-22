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

// CIM_ServiceServiceDependency struct
type CIM_ServiceServiceDependency struct {
	*CIM_Dependency

	//
	TypeOfDependency uint16
}

func NewCIM_ServiceServiceDependencyEx1(instance *cim.WmiInstance) (newInstance *CIM_ServiceServiceDependency, err error) {
	tmp, err := NewCIM_DependencyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ServiceServiceDependency{
		CIM_Dependency: tmp,
	}
	return
}

func NewCIM_ServiceServiceDependencyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ServiceServiceDependency, err error) {
	tmp, err := NewCIM_DependencyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ServiceServiceDependency{
		CIM_Dependency: tmp,
	}
	return
}

// SetTypeOfDependency sets the value of TypeOfDependency for the instance
func (instance *CIM_ServiceServiceDependency) SetPropertyTypeOfDependency(value uint16) (err error) {
	return instance.SetProperty("TypeOfDependency", (value))
}

// GetTypeOfDependency gets the value of TypeOfDependency for the instance
func (instance *CIM_ServiceServiceDependency) GetPropertyTypeOfDependency() (value uint16, err error) {
	retValue, err := instance.GetProperty("TypeOfDependency")
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
