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

// MSFT_NetServiceStopControlSuccess struct
type MSFT_NetServiceStopControlSuccess struct {
	*MSFT_SCMEventLogEvent

	//
	Comment string

	//
	Control string

	//
	Reason string

	//
	ReasonText string

	//
	Service string

	//
	sid string
}

func NewMSFT_NetServiceStopControlSuccessEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetServiceStopControlSuccess, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceStopControlSuccess{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetServiceStopControlSuccessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetServiceStopControlSuccess, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceStopControlSuccess{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetComment sets the value of Comment for the instance
func (instance *MSFT_NetServiceStopControlSuccess) SetPropertyComment(value string) (err error) {
	return instance.SetProperty("Comment", (value))
}

// GetComment gets the value of Comment for the instance
func (instance *MSFT_NetServiceStopControlSuccess) GetPropertyComment() (value string, err error) {
	retValue, err := instance.GetProperty("Comment")
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

// SetControl sets the value of Control for the instance
func (instance *MSFT_NetServiceStopControlSuccess) SetPropertyControl(value string) (err error) {
	return instance.SetProperty("Control", (value))
}

// GetControl gets the value of Control for the instance
func (instance *MSFT_NetServiceStopControlSuccess) GetPropertyControl() (value string, err error) {
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

// SetReason sets the value of Reason for the instance
func (instance *MSFT_NetServiceStopControlSuccess) SetPropertyReason(value string) (err error) {
	return instance.SetProperty("Reason", (value))
}

// GetReason gets the value of Reason for the instance
func (instance *MSFT_NetServiceStopControlSuccess) GetPropertyReason() (value string, err error) {
	retValue, err := instance.GetProperty("Reason")
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

// SetReasonText sets the value of ReasonText for the instance
func (instance *MSFT_NetServiceStopControlSuccess) SetPropertyReasonText(value string) (err error) {
	return instance.SetProperty("ReasonText", (value))
}

// GetReasonText gets the value of ReasonText for the instance
func (instance *MSFT_NetServiceStopControlSuccess) GetPropertyReasonText() (value string, err error) {
	retValue, err := instance.GetProperty("ReasonText")
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
func (instance *MSFT_NetServiceStopControlSuccess) SetPropertyService(value string) (err error) {
	return instance.SetProperty("Service", (value))
}

// GetService gets the value of Service for the instance
func (instance *MSFT_NetServiceStopControlSuccess) GetPropertyService() (value string, err error) {
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
func (instance *MSFT_NetServiceStopControlSuccess) SetPropertysid(value string) (err error) {
	return instance.SetProperty("sid", (value))
}

// Getsid gets the value of sid for the instance
func (instance *MSFT_NetServiceStopControlSuccess) GetPropertysid() (value string, err error) {
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
