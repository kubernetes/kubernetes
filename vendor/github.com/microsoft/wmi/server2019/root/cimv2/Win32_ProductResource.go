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

// Win32_ProductResource struct
type Win32_ProductResource struct {
	*cim.WmiInstance

	//
	Product Win32_Product

	//
	Resource Win32_MSIResource
}

func NewWin32_ProductResourceEx1(instance *cim.WmiInstance) (newInstance *Win32_ProductResource, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ProductResource{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ProductResourceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ProductResource, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ProductResource{
		WmiInstance: tmp,
	}
	return
}

// SetProduct sets the value of Product for the instance
func (instance *Win32_ProductResource) SetPropertyProduct(value Win32_Product) (err error) {
	return instance.SetProperty("Product", (value))
}

// GetProduct gets the value of Product for the instance
func (instance *Win32_ProductResource) GetPropertyProduct() (value Win32_Product, err error) {
	retValue, err := instance.GetProperty("Product")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_Product)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_Product is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_Product(valuetmp)

	return
}

// SetResource sets the value of Resource for the instance
func (instance *Win32_ProductResource) SetPropertyResource(value Win32_MSIResource) (err error) {
	return instance.SetProperty("Resource", (value))
}

// GetResource gets the value of Resource for the instance
func (instance *Win32_ProductResource) GetPropertyResource() (value Win32_MSIResource, err error) {
	retValue, err := instance.GetProperty("Resource")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_MSIResource)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_MSIResource is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_MSIResource(valuetmp)

	return
}
