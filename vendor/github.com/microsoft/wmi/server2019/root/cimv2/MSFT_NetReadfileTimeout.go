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

// MSFT_NetReadfileTimeout struct
type MSFT_NetReadfileTimeout struct {
	*MSFT_SCMEventLogEvent

	//
	Milliseconds uint32
}

func NewMSFT_NetReadfileTimeoutEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetReadfileTimeout, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetReadfileTimeout{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetReadfileTimeoutEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetReadfileTimeout, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetReadfileTimeout{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetMilliseconds sets the value of Milliseconds for the instance
func (instance *MSFT_NetReadfileTimeout) SetPropertyMilliseconds(value uint32) (err error) {
	return instance.SetProperty("Milliseconds", (value))
}

// GetMilliseconds gets the value of Milliseconds for the instance
func (instance *MSFT_NetReadfileTimeout) GetPropertyMilliseconds() (value uint32, err error) {
	retValue, err := instance.GetProperty("Milliseconds")
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
