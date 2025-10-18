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

// Win32_ComputerShutdownEvent struct
type Win32_ComputerShutdownEvent struct {
	*Win32_ComputerSystemEvent

	//
	Type uint32
}

func NewWin32_ComputerShutdownEventEx1(instance *cim.WmiInstance) (newInstance *Win32_ComputerShutdownEvent, err error) {
	tmp, err := NewWin32_ComputerSystemEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerShutdownEvent{
		Win32_ComputerSystemEvent: tmp,
	}
	return
}

func NewWin32_ComputerShutdownEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ComputerShutdownEvent, err error) {
	tmp, err := NewWin32_ComputerSystemEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerShutdownEvent{
		Win32_ComputerSystemEvent: tmp,
	}
	return
}

// SetType sets the value of Type for the instance
func (instance *Win32_ComputerShutdownEvent) SetPropertyType(value uint32) (err error) {
	return instance.SetProperty("Type", (value))
}

// GetType gets the value of Type for the instance
func (instance *Win32_ComputerShutdownEvent) GetPropertyType() (value uint32, err error) {
	retValue, err := instance.GetProperty("Type")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}
