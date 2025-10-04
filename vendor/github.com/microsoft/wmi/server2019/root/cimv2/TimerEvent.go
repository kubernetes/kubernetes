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

// __TimerEvent struct
type __TimerEvent struct {
	*__Event

	//
	NumFirings uint32

	//
	TimerId string
}

func New__TimerEventEx1(instance *cim.WmiInstance) (newInstance *__TimerEvent, err error) {
	tmp, err := New__EventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__TimerEvent{
		__Event: tmp,
	}
	return
}

func New__TimerEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__TimerEvent, err error) {
	tmp, err := New__EventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__TimerEvent{
		__Event: tmp,
	}
	return
}

// SetNumFirings sets the value of NumFirings for the instance
func (instance *__TimerEvent) SetPropertyNumFirings(value uint32) (err error) {
	return instance.SetProperty("NumFirings", (value))
}

// GetNumFirings gets the value of NumFirings for the instance
func (instance *__TimerEvent) GetPropertyNumFirings() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumFirings")
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

// SetTimerId sets the value of TimerId for the instance
func (instance *__TimerEvent) SetPropertyTimerId(value string) (err error) {
	return instance.SetProperty("TimerId", (value))
}

// GetTimerId gets the value of TimerId for the instance
func (instance *__TimerEvent) GetPropertyTimerId() (value string, err error) {
	retValue, err := instance.GetProperty("TimerId")
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
