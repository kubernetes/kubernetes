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

// Win32_PowerManagementEvent struct
type Win32_PowerManagementEvent struct {
	*__ExtrinsicEvent

	//
	EventType uint16

	//
	OEMEventCode uint16
}

func NewWin32_PowerManagementEventEx1(instance *cim.WmiInstance) (newInstance *Win32_PowerManagementEvent, err error) {
	tmp, err := New__ExtrinsicEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PowerManagementEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

func NewWin32_PowerManagementEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PowerManagementEvent, err error) {
	tmp, err := New__ExtrinsicEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PowerManagementEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

// SetEventType sets the value of EventType for the instance
func (instance *Win32_PowerManagementEvent) SetPropertyEventType(value uint16) (err error) {
	return instance.SetProperty("EventType", (value))
}

// GetEventType gets the value of EventType for the instance
func (instance *Win32_PowerManagementEvent) GetPropertyEventType() (value uint16, err error) {
	retValue, err := instance.GetProperty("EventType")
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

// SetOEMEventCode sets the value of OEMEventCode for the instance
func (instance *Win32_PowerManagementEvent) SetPropertyOEMEventCode(value uint16) (err error) {
	return instance.SetProperty("OEMEventCode", (value))
}

// GetOEMEventCode gets the value of OEMEventCode for the instance
func (instance *Win32_PowerManagementEvent) GetPropertyOEMEventCode() (value uint16, err error) {
	retValue, err := instance.GetProperty("OEMEventCode")
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
