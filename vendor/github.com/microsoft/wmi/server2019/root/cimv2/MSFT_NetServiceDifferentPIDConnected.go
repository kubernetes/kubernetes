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

// MSFT_NetServiceDifferentPIDConnected struct
type MSFT_NetServiceDifferentPIDConnected struct {
	*MSFT_SCMEventLogEvent

	//
	ActualPID uint32

	//
	ExpectedPID uint32

	//
	Service string
}

func NewMSFT_NetServiceDifferentPIDConnectedEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetServiceDifferentPIDConnected, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceDifferentPIDConnected{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetServiceDifferentPIDConnectedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetServiceDifferentPIDConnected, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceDifferentPIDConnected{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetActualPID sets the value of ActualPID for the instance
func (instance *MSFT_NetServiceDifferentPIDConnected) SetPropertyActualPID(value uint32) (err error) {
	return instance.SetProperty("ActualPID", (value))
}

// GetActualPID gets the value of ActualPID for the instance
func (instance *MSFT_NetServiceDifferentPIDConnected) GetPropertyActualPID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ActualPID")
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

// SetExpectedPID sets the value of ExpectedPID for the instance
func (instance *MSFT_NetServiceDifferentPIDConnected) SetPropertyExpectedPID(value uint32) (err error) {
	return instance.SetProperty("ExpectedPID", (value))
}

// GetExpectedPID gets the value of ExpectedPID for the instance
func (instance *MSFT_NetServiceDifferentPIDConnected) GetPropertyExpectedPID() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExpectedPID")
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

// SetService sets the value of Service for the instance
func (instance *MSFT_NetServiceDifferentPIDConnected) SetPropertyService(value string) (err error) {
	return instance.SetProperty("Service", (value))
}

// GetService gets the value of Service for the instance
func (instance *MSFT_NetServiceDifferentPIDConnected) GetPropertyService() (value string, err error) {
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
