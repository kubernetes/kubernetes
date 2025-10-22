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

// Win32_ParallelPort struct
type Win32_ParallelPort struct {
	*CIM_ParallelController

	//
	OSAutoDiscovered bool
}

func NewWin32_ParallelPortEx1(instance *cim.WmiInstance) (newInstance *Win32_ParallelPort, err error) {
	tmp, err := NewCIM_ParallelControllerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ParallelPort{
		CIM_ParallelController: tmp,
	}
	return
}

func NewWin32_ParallelPortEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ParallelPort, err error) {
	tmp, err := NewCIM_ParallelControllerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ParallelPort{
		CIM_ParallelController: tmp,
	}
	return
}

// SetOSAutoDiscovered sets the value of OSAutoDiscovered for the instance
func (instance *Win32_ParallelPort) SetPropertyOSAutoDiscovered(value bool) (err error) {
	return instance.SetProperty("OSAutoDiscovered", (value))
}

// GetOSAutoDiscovered gets the value of OSAutoDiscovered for the instance
func (instance *Win32_ParallelPort) GetPropertyOSAutoDiscovered() (value bool, err error) {
	retValue, err := instance.GetProperty("OSAutoDiscovered")
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
