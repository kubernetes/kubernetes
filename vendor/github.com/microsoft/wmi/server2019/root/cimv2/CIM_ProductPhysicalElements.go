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

// CIM_ProductPhysicalElements struct
type CIM_ProductPhysicalElements struct {
	*cim.WmiInstance

	//
	Component CIM_PhysicalElement

	//
	Product CIM_Product
}

func NewCIM_ProductPhysicalElementsEx1(instance *cim.WmiInstance) (newInstance *CIM_ProductPhysicalElements, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_ProductPhysicalElements{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ProductPhysicalElementsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ProductPhysicalElements, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ProductPhysicalElements{
		WmiInstance: tmp,
	}
	return
}

// SetComponent sets the value of Component for the instance
func (instance *CIM_ProductPhysicalElements) SetPropertyComponent(value CIM_PhysicalElement) (err error) {
	return instance.SetProperty("Component", (value))
}

// GetComponent gets the value of Component for the instance
func (instance *CIM_ProductPhysicalElements) GetPropertyComponent() (value CIM_PhysicalElement, err error) {
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

// SetProduct sets the value of Product for the instance
func (instance *CIM_ProductPhysicalElements) SetPropertyProduct(value CIM_Product) (err error) {
	return instance.SetProperty("Product", (value))
}

// GetProduct gets the value of Product for the instance
func (instance *CIM_ProductPhysicalElements) GetPropertyProduct() (value CIM_Product, err error) {
	retValue, err := instance.GetProperty("Product")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Product)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Product is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Product(valuetmp)

	return
}
