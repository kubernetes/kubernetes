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

// Win32_ImplementedCategory struct
type Win32_ImplementedCategory struct {
	*cim.WmiInstance

	//
	Category Win32_ComponentCategory

	//
	Component Win32_ClassicCOMClass
}

func NewWin32_ImplementedCategoryEx1(instance *cim.WmiInstance) (newInstance *Win32_ImplementedCategory, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ImplementedCategory{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ImplementedCategoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ImplementedCategory, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ImplementedCategory{
		WmiInstance: tmp,
	}
	return
}

// SetCategory sets the value of Category for the instance
func (instance *Win32_ImplementedCategory) SetPropertyCategory(value Win32_ComponentCategory) (err error) {
	return instance.SetProperty("Category", (value))
}

// GetCategory gets the value of Category for the instance
func (instance *Win32_ImplementedCategory) GetPropertyCategory() (value Win32_ComponentCategory, err error) {
	retValue, err := instance.GetProperty("Category")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_ComponentCategory)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_ComponentCategory is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_ComponentCategory(valuetmp)

	return
}

// SetComponent sets the value of Component for the instance
func (instance *Win32_ImplementedCategory) SetPropertyComponent(value Win32_ClassicCOMClass) (err error) {
	return instance.SetProperty("Component", (value))
}

// GetComponent gets the value of Component for the instance
func (instance *Win32_ImplementedCategory) GetPropertyComponent() (value Win32_ClassicCOMClass, err error) {
	retValue, err := instance.GetProperty("Component")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_ClassicCOMClass)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_ClassicCOMClass is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_ClassicCOMClass(valuetmp)

	return
}
