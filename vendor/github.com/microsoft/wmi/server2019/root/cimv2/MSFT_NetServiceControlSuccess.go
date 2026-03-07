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

// MSFT_NetServiceControlSuccess struct
type MSFT_NetServiceControlSuccess struct {
	*MSFT_SCMEventLogEvent

	//
	Control string

	//
	Service string

	//
	sid string
}

func NewMSFT_NetServiceControlSuccessEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetServiceControlSuccess, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceControlSuccess{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetServiceControlSuccessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetServiceControlSuccess, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceControlSuccess{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetControl sets the value of Control for the instance
func (instance *MSFT_NetServiceControlSuccess) SetPropertyControl(value string) (err error) {
	return instance.SetProperty("Control", (value))
}

// GetControl gets the value of Control for the instance
func (instance *MSFT_NetServiceControlSuccess) GetPropertyControl() (value string, err error) {
	retValue, err := instance.GetProperty("Control")
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

// SetService sets the value of Service for the instance
func (instance *MSFT_NetServiceControlSuccess) SetPropertyService(value string) (err error) {
	return instance.SetProperty("Service", (value))
}

// GetService gets the value of Service for the instance
func (instance *MSFT_NetServiceControlSuccess) GetPropertyService() (value string, err error) {
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

// Setsid sets the value of sid for the instance
func (instance *MSFT_NetServiceControlSuccess) SetPropertysid(value string) (err error) {
	return instance.SetProperty("sid", (value))
}

// Getsid gets the value of sid for the instance
func (instance *MSFT_NetServiceControlSuccess) GetPropertysid() (value string, err error) {
	retValue, err := instance.GetProperty("sid")
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
