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

// MSFT_NetServiceStartFailedII struct
type MSFT_NetServiceStartFailedII struct {
	*MSFT_SCMEventLogEvent

	//
	DependedOnService string

	//
	Error uint32

	//
	Service string
}

func NewMSFT_NetServiceStartFailedIIEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetServiceStartFailedII, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceStartFailedII{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetServiceStartFailedIIEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetServiceStartFailedII, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceStartFailedII{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetDependedOnService sets the value of DependedOnService for the instance
func (instance *MSFT_NetServiceStartFailedII) SetPropertyDependedOnService(value string) (err error) {
	return instance.SetProperty("DependedOnService", (value))
}

// GetDependedOnService gets the value of DependedOnService for the instance
func (instance *MSFT_NetServiceStartFailedII) GetPropertyDependedOnService() (value string, err error) {
	retValue, err := instance.GetProperty("DependedOnService")
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

// SetError sets the value of Error for the instance
func (instance *MSFT_NetServiceStartFailedII) SetPropertyError(value uint32) (err error) {
	return instance.SetProperty("Error", (value))
}

// GetError gets the value of Error for the instance
func (instance *MSFT_NetServiceStartFailedII) GetPropertyError() (value uint32, err error) {
	retValue, err := instance.GetProperty("Error")
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
func (instance *MSFT_NetServiceStartFailedII) SetPropertyService(value string) (err error) {
	return instance.SetProperty("Service", (value))
}

// GetService gets the value of Service for the instance
func (instance *MSFT_NetServiceStartFailedII) GetPropertyService() (value string, err error) {
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
