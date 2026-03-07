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

// CIM_ProductProductDependency struct
type CIM_ProductProductDependency struct {
	*cim.WmiInstance

	//
	DependentProduct CIM_Product

	//
	RequiredProduct CIM_Product

	//
	TypeOfDependency uint16
}

func NewCIM_ProductProductDependencyEx1(instance *cim.WmiInstance) (newInstance *CIM_ProductProductDependency, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_ProductProductDependency{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_ProductProductDependencyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ProductProductDependency, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ProductProductDependency{
		WmiInstance: tmp,
	}
	return
}

// SetDependentProduct sets the value of DependentProduct for the instance
func (instance *CIM_ProductProductDependency) SetPropertyDependentProduct(value CIM_Product) (err error) {
	return instance.SetProperty("DependentProduct", (value))
}

// GetDependentProduct gets the value of DependentProduct for the instance
func (instance *CIM_ProductProductDependency) GetPropertyDependentProduct() (value CIM_Product, err error) {
	retValue, err := instance.GetProperty("DependentProduct")
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

// SetRequiredProduct sets the value of RequiredProduct for the instance
func (instance *CIM_ProductProductDependency) SetPropertyRequiredProduct(value CIM_Product) (err error) {
	return instance.SetProperty("RequiredProduct", (value))
}

// GetRequiredProduct gets the value of RequiredProduct for the instance
func (instance *CIM_ProductProductDependency) GetPropertyRequiredProduct() (value CIM_Product, err error) {
	retValue, err := instance.GetProperty("RequiredProduct")
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

// SetTypeOfDependency sets the value of TypeOfDependency for the instance
func (instance *CIM_ProductProductDependency) SetPropertyTypeOfDependency(value uint16) (err error) {
	return instance.SetProperty("TypeOfDependency", (value))
}

// GetTypeOfDependency gets the value of TypeOfDependency for the instance
func (instance *CIM_ProductProductDependency) GetPropertyTypeOfDependency() (value uint16, err error) {
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
