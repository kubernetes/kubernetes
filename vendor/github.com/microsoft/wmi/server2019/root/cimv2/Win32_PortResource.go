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

// Win32_PortResource struct
type Win32_PortResource struct {
	*Win32_SystemMemoryResource

	//
	Alias bool
}

func NewWin32_PortResourceEx1(instance *cim.WmiInstance) (newInstance *Win32_PortResource, err error) {
	tmp, err := NewWin32_SystemMemoryResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PortResource{
		Win32_SystemMemoryResource: tmp,
	}
	return
}

func NewWin32_PortResourceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PortResource, err error) {
	tmp, err := NewWin32_SystemMemoryResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PortResource{
		Win32_SystemMemoryResource: tmp,
	}
	return
}

// SetAlias sets the value of Alias for the instance
func (instance *Win32_PortResource) SetPropertyAlias(value bool) (err error) {
	return instance.SetProperty("Alias", (value))
}

// GetAlias gets the value of Alias for the instance
func (instance *Win32_PortResource) GetPropertyAlias() (value bool, err error) {
	retValue, err := instance.GetProperty("Alias")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}
