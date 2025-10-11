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

// Win32_SoftwareElementResource struct
type Win32_SoftwareElementResource struct {
	*Win32_ManagedSystemElementResource

	//
	Element Win32_SoftwareElement

	//
	Setting Win32_MSIResource
}

func NewWin32_SoftwareElementResourceEx1(instance *cim.WmiInstance) (newInstance *Win32_SoftwareElementResource, err error) {
	tmp, err := NewWin32_ManagedSystemElementResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareElementResource{
		Win32_ManagedSystemElementResource: tmp,
	}
	return
}

func NewWin32_SoftwareElementResourceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SoftwareElementResource, err error) {
	tmp, err := NewWin32_ManagedSystemElementResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SoftwareElementResource{
		Win32_ManagedSystemElementResource: tmp,
	}
	return
}

// SetElement sets the value of Element for the instance
func (instance *Win32_SoftwareElementResource) SetPropertyElement(value Win32_SoftwareElement) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *Win32_SoftwareElementResource) GetPropertyElement() (value Win32_SoftwareElement, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_SoftwareElement)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_SoftwareElement is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_SoftwareElement(valuetmp)

	return
}

// SetSetting sets the value of Setting for the instance
func (instance *Win32_SoftwareElementResource) SetPropertySetting(value Win32_MSIResource) (err error) {
	return instance.SetProperty("Setting", (value))
}

// GetSetting gets the value of Setting for the instance
func (instance *Win32_SoftwareElementResource) GetPropertySetting() (value Win32_MSIResource, err error) {
	retValue, err := instance.GetProperty("Setting")
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
