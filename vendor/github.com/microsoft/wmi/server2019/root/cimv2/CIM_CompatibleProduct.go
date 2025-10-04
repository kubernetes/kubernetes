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

// CIM_CompatibleProduct struct
type CIM_CompatibleProduct struct {
	*cim.WmiInstance

	//
	CompatibilityDescription string

	//
	CompatibleProduct CIM_Product

	//
	Product CIM_Product
}

func NewCIM_CompatibleProductEx1(instance *cim.WmiInstance) (newInstance *CIM_CompatibleProduct, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_CompatibleProduct{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_CompatibleProductEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_CompatibleProduct, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_CompatibleProduct{
		WmiInstance: tmp,
	}
	return
}

// SetCompatibilityDescription sets the value of CompatibilityDescription for the instance
func (instance *CIM_CompatibleProduct) SetPropertyCompatibilityDescription(value string) (err error) {
	return instance.SetProperty("CompatibilityDescription", (value))
}

// GetCompatibilityDescription gets the value of CompatibilityDescription for the instance
func (instance *CIM_CompatibleProduct) GetPropertyCompatibilityDescription() (value string, err error) {
	retValue, err := instance.GetProperty("CompatibilityDescription")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetCompatibleProduct sets the value of CompatibleProduct for the instance
func (instance *CIM_CompatibleProduct) SetPropertyCompatibleProduct(value CIM_Product) (err error) {
	return instance.SetProperty("CompatibleProduct", (value))
}

// GetCompatibleProduct gets the value of CompatibleProduct for the instance
func (instance *CIM_CompatibleProduct) GetPropertyCompatibleProduct() (value CIM_Product, err error) {
	retValue, err := instance.GetProperty("CompatibleProduct")
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

// SetProduct sets the value of Product for the instance
func (instance *CIM_CompatibleProduct) SetPropertyProduct(value CIM_Product) (err error) {
	return instance.SetProperty("Product", (value))
}

// GetProduct gets the value of Product for the instance
func (instance *CIM_CompatibleProduct) GetPropertyProduct() (value CIM_Product, err error) {
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
