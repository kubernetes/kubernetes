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

// Win32_PatchPackage struct
type Win32_PatchPackage struct {
	*Win32_MSIResource

	//
	PatchID string

	//
	ProductCode string
}

func NewWin32_PatchPackageEx1(instance *cim.WmiInstance) (newInstance *Win32_PatchPackage, err error) {
	tmp, err := NewWin32_MSIResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PatchPackage{
		Win32_MSIResource: tmp,
	}
	return
}

func NewWin32_PatchPackageEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PatchPackage, err error) {
	tmp, err := NewWin32_MSIResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PatchPackage{
		Win32_MSIResource: tmp,
	}
	return
}

// SetPatchID sets the value of PatchID for the instance
func (instance *Win32_PatchPackage) SetPropertyPatchID(value string) (err error) {
	return instance.SetProperty("PatchID", (value))
}

// GetPatchID gets the value of PatchID for the instance
func (instance *Win32_PatchPackage) GetPropertyPatchID() (value string, err error) {
	retValue, err := instance.GetProperty("PatchID")
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

// SetProductCode sets the value of ProductCode for the instance
func (instance *Win32_PatchPackage) SetPropertyProductCode(value string) (err error) {
	return instance.SetProperty("ProductCode", (value))
}

// GetProductCode gets the value of ProductCode for the instance
func (instance *Win32_PatchPackage) GetPropertyProductCode() (value string, err error) {
	retValue, err := instance.GetProperty("ProductCode")
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
