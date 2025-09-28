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

// MSFT_NetServiceStartTypeChanged struct
type MSFT_NetServiceStartTypeChanged struct {
	*MSFT_SCMEventLogEvent

	//
	NewStartType string

	//
	OldStartType string

	//
	Service string

	//
	sid string
}

func NewMSFT_NetServiceStartTypeChangedEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetServiceStartTypeChanged, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceStartTypeChanged{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetServiceStartTypeChangedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetServiceStartTypeChanged, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceStartTypeChanged{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetNewStartType sets the value of NewStartType for the instance
func (instance *MSFT_NetServiceStartTypeChanged) SetPropertyNewStartType(value string) (err error) {
	return instance.SetProperty("NewStartType", (value))
}

// GetNewStartType gets the value of NewStartType for the instance
func (instance *MSFT_NetServiceStartTypeChanged) GetPropertyNewStartType() (value string, err error) {
	retValue, err := instance.GetProperty("NewStartType")
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

// SetOldStartType sets the value of OldStartType for the instance
func (instance *MSFT_NetServiceStartTypeChanged) SetPropertyOldStartType(value string) (err error) {
	return instance.SetProperty("OldStartType", (value))
}

// GetOldStartType gets the value of OldStartType for the instance
func (instance *MSFT_NetServiceStartTypeChanged) GetPropertyOldStartType() (value string, err error) {
	retValue, err := instance.GetProperty("OldStartType")
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
func (instance *MSFT_NetServiceStartTypeChanged) SetPropertyService(value string) (err error) {
	return instance.SetProperty("Service", (value))
}

// GetService gets the value of Service for the instance
func (instance *MSFT_NetServiceStartTypeChanged) GetPropertyService() (value string, err error) {
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
func (instance *MSFT_NetServiceStartTypeChanged) SetPropertysid(value string) (err error) {
	return instance.SetProperty("sid", (value))
}

// Getsid gets the value of sid for the instance
func (instance *MSFT_NetServiceStartTypeChanged) GetPropertysid() (value string, err error) {
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
