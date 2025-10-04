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

// CIM_FRUIncludesProduct struct
type CIM_FRUIncludesProduct struct {
	*cim.WmiInstance

	//
	Component CIM_Product

	//
	FRU CIM_FRU
}

func NewCIM_FRUIncludesProductEx1(instance *cim.WmiInstance) (newInstance *CIM_FRUIncludesProduct, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_FRUIncludesProduct{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_FRUIncludesProductEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_FRUIncludesProduct, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_FRUIncludesProduct{
		WmiInstance: tmp,
	}
	return
}

// SetComponent sets the value of Component for the instance
func (instance *CIM_FRUIncludesProduct) SetPropertyComponent(value CIM_Product) (err error) {
	return instance.SetProperty("Component", (value))
}

// GetComponent gets the value of Component for the instance
func (instance *CIM_FRUIncludesProduct) GetPropertyComponent() (value CIM_Product, err error) {
	retValue, err := instance.GetProperty("Component")
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

// SetFRU sets the value of FRU for the instance
func (instance *CIM_FRUIncludesProduct) SetPropertyFRU(value CIM_FRU) (err error) {
	return instance.SetProperty("FRU", (value))
}

// GetFRU gets the value of FRU for the instance
func (instance *CIM_FRUIncludesProduct) GetPropertyFRU() (value CIM_FRU, err error) {
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
