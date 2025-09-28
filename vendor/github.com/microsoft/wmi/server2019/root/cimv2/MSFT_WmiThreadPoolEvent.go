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

// MSFT_WmiThreadPoolEvent struct
type MSFT_WmiThreadPoolEvent struct {
	*MSFT_WmiEssEvent

	//
	ThreadId uint32
}

func NewMSFT_WmiThreadPoolEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_WmiThreadPoolEvent, err error) {
	tmp, err := NewMSFT_WmiEssEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiThreadPoolEvent{
		MSFT_WmiEssEvent: tmp,
	}
	return
}

func NewMSFT_WmiThreadPoolEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WmiThreadPoolEvent, err error) {
	tmp, err := NewMSFT_WmiEssEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WmiThreadPoolEvent{
		MSFT_WmiEssEvent: tmp,
	}
	return
}

// SetThreadId sets the value of ThreadId for the instance
func (instance *MSFT_WmiThreadPoolEvent) SetPropertyThreadId(value uint32) (err error) {
	return instance.SetProperty("ThreadId", (value))
}

// GetThreadId gets the value of ThreadId for the instance
func (instance *MSFT_WmiThreadPoolEvent) GetPropertyThreadId() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThreadId")
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
