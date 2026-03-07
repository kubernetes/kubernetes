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

// MSFT_NetBootSystemDriversFailed struct
type MSFT_NetBootSystemDriversFailed struct {
	*MSFT_SCMEventLogEvent

	//
	DriverList string
}

func NewMSFT_NetBootSystemDriversFailedEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetBootSystemDriversFailed, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetBootSystemDriversFailed{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetBootSystemDriversFailedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetBootSystemDriversFailed, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetBootSystemDriversFailed{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetDriverList sets the value of DriverList for the instance
func (instance *MSFT_NetBootSystemDriversFailed) SetPropertyDriverList(value string) (err error) {
	return instance.SetProperty("DriverList", (value))
}

// GetDriverList gets the value of DriverList for the instance
func (instance *MSFT_NetBootSystemDriversFailed) GetPropertyDriverList() (value string, err error) {
	retValue, err := instance.GetProperty("DriverList")
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
