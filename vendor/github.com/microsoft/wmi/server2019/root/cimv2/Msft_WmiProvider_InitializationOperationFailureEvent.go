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

// Msft_WmiProvider_InitializationOperationFailureEvent struct
type Msft_WmiProvider_InitializationOperationFailureEvent struct {
	*Msft_WmiProvider_OperationEvent

	//
	ResultCode uint32
}

func NewMsft_WmiProvider_InitializationOperationFailureEventEx1(instance *cim.WmiInstance) (newInstance *Msft_WmiProvider_InitializationOperationFailureEvent, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_InitializationOperationFailureEvent{
		Msft_WmiProvider_OperationEvent: tmp,
	}
	return
}

func NewMsft_WmiProvider_InitializationOperationFailureEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Msft_WmiProvider_InitializationOperationFailureEvent, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_InitializationOperationFailureEvent{
		Msft_WmiProvider_OperationEvent: tmp,
	}
	return
}

// SetResultCode sets the value of ResultCode for the instance
func (instance *Msft_WmiProvider_InitializationOperationFailureEvent) SetPropertyResultCode(value uint32) (err error) {
	return instance.SetProperty("ResultCode", (value))
}

// GetResultCode gets the value of ResultCode for the instance
func (instance *Msft_WmiProvider_InitializationOperationFailureEvent) GetPropertyResultCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("ResultCode")
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
