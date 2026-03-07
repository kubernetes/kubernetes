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

// MSFT_NetServiceCrashNoAction struct
type MSFT_NetServiceCrashNoAction struct {
	*MSFT_SCMEventLogEvent

	//
	Service string

	//
	TimesFailed uint32
}

func NewMSFT_NetServiceCrashNoActionEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetServiceCrashNoAction, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceCrashNoAction{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetServiceCrashNoActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetServiceCrashNoAction, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceCrashNoAction{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetService sets the value of Service for the instance
func (instance *MSFT_NetServiceCrashNoAction) SetPropertyService(value string) (err error) {
	return instance.SetProperty("Service", (value))
}

// GetService gets the value of Service for the instance
func (instance *MSFT_NetServiceCrashNoAction) GetPropertyService() (value string, err error) {
	retValue, err := instance.GetProperty("Service")
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

// SetTimesFailed sets the value of TimesFailed for the instance
func (instance *MSFT_NetServiceCrashNoAction) SetPropertyTimesFailed(value uint32) (err error) {
	return instance.SetProperty("TimesFailed", (value))
}

// GetTimesFailed gets the value of TimesFailed for the instance
func (instance *MSFT_NetServiceCrashNoAction) GetPropertyTimesFailed() (value uint32, err error) {
	retValue, err := instance.GetProperty("TimesFailed")
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
