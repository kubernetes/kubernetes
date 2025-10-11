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

// Win32_ComputerSystemEvent struct
type Win32_ComputerSystemEvent struct {
	*__ExtrinsicEvent

	//
	MachineName string
}

func NewWin32_ComputerSystemEventEx1(instance *cim.WmiInstance) (newInstance *Win32_ComputerSystemEvent, err error) {
	tmp, err := New__ExtrinsicEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystemEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

func NewWin32_ComputerSystemEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ComputerSystemEvent, err error) {
	tmp, err := New__ExtrinsicEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystemEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

// SetMachineName sets the value of MachineName for the instance
func (instance *Win32_ComputerSystemEvent) SetPropertyMachineName(value string) (err error) {
	return instance.SetProperty("MachineName", (value))
}

// GetMachineName gets the value of MachineName for the instance
func (instance *Win32_ComputerSystemEvent) GetPropertyMachineName() (value string, err error) {
	retValue, err := instance.GetProperty("MachineName")
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
