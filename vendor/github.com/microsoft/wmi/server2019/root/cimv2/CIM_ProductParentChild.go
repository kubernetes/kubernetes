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

// CIM_ProductParentChild struct
type CIM_ProductParentChild struct {
	*cim.WmiInstance

	//
	Child CIM_Product

	//
	Parent CIM_Product
}

func NewCIM_ProductParentChildEx1(instance *cim.WmiInstance) (newInstance *CIM_ProductParentChild, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_ProductParentChild{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ProductParentChildEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ProductParentChild, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ProductParentChild{
		WmiInstance: tmp,
	}
	return
}

// SetChild sets the value of Child for the instance
func (instance *CIM_ProductParentChild) SetPropertyChild(value CIM_Product) (err error) {
	return instance.SetProperty("Child", (value))
}

// GetChild gets the value of Child for the instance
func (instance *CIM_ProductParentChild) GetPropertyChild() (value CIM_Product, err error) {
	retValue, err := instance.GetProperty("Child")
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

// SetParent sets the value of Parent for the instance
func (instance *CIM_ProductParentChild) SetPropertyParent(value CIM_Product) (err error) {
	return instance.SetProperty("Parent", (value))
}

// GetParent gets the value of Parent for the instance
func (instance *CIM_ProductParentChild) GetPropertyParent() (value CIM_Product, err error) {
	retValue, err := instance.GetProperty("Parent")
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
