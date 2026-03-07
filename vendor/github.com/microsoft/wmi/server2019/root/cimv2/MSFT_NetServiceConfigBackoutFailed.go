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

// MSFT_NetServiceConfigBackoutFailed struct
type MSFT_NetServiceConfigBackoutFailed struct {
	*MSFT_SCMEventLogEvent

	//
	ConfigField string

	//
	Service string
}

func NewMSFT_NetServiceConfigBackoutFailedEx1(instance *cim.WmiInstance) (newInstance *MSFT_NetServiceConfigBackoutFailed, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceConfigBackoutFailed{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

func NewMSFT_NetServiceConfigBackoutFailedEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_NetServiceConfigBackoutFailed, err error) {
	tmp, err := NewMSFT_SCMEventLogEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_NetServiceConfigBackoutFailed{
		MSFT_SCMEventLogEvent: tmp,
	}
	return
}

// SetConfigField sets the value of ConfigField for the instance
func (instance *MSFT_NetServiceConfigBackoutFailed) SetPropertyConfigField(value string) (err error) {
	return instance.SetProperty("ConfigField", (value))
}

// GetConfigField gets the value of ConfigField for the instance
func (instance *MSFT_NetServiceConfigBackoutFailed) GetPropertyConfigField() (value string, err error) {
	retValue, err := instance.GetProperty("ConfigField")
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
func (instance *MSFT_NetServiceConfigBackoutFailed) SetPropertyService(value string) (err error) {
	return instance.SetProperty("Service", (value))
}

// GetService gets the value of Service for the instance
func (instance *MSFT_NetServiceConfigBackoutFailed) GetPropertyService() (value string, err error) {
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
