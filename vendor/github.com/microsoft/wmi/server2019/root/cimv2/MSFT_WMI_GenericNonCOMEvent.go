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

// MSFT_WMI_GenericNonCOMEvent struct
type MSFT_WMI_GenericNonCOMEvent struct {
	*__ExtrinsicEvent

	//
	ProcessId uint32

	//
	PropertyNames []string

	//
	PropertyValues []string

	//
	ProviderName string
}

func NewMSFT_WMI_GenericNonCOMEventEx1(instance *cim.WmiInstance) (newInstance *MSFT_WMI_GenericNonCOMEvent, err error) {
	tmp, err := New__ExtrinsicEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &MSFT_WMI_GenericNonCOMEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

func NewMSFT_WMI_GenericNonCOMEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_WMI_GenericNonCOMEvent, err error) {
	tmp, err := New__ExtrinsicEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_WMI_GenericNonCOMEvent{
		__ExtrinsicEvent: tmp,
	}
	return
}

// SetProcessId sets the value of ProcessId for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) SetPropertyProcessId(value uint32) (err error) {
	return instance.SetProperty("ProcessId", (value))
}

// GetProcessId gets the value of ProcessId for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) GetPropertyProcessId() (value uint32, err error) {
	retValue, err := instance.GetProperty("ProcessId")
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

// SetPropertyNames sets the value of PropertyNames for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) SetPropertyPropertyNames(value []string) (err error) {
	return instance.SetProperty("PropertyNames", (value))
}

// GetPropertyNames gets the value of PropertyNames for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) GetPropertyPropertyNames() (value []string, err error) {
	retValue, err := instance.GetProperty("PropertyNames")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetPropertyValues sets the value of PropertyValues for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) SetPropertyPropertyValues(value []string) (err error) {
	return instance.SetProperty("PropertyValues", (value))
}

// GetPropertyValues gets the value of PropertyValues for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) GetPropertyPropertyValues() (value []string, err error) {
	retValue, err := instance.GetProperty("PropertyValues")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetProviderName sets the value of ProviderName for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) SetPropertyProviderName(value string) (err error) {
	return instance.SetProperty("ProviderName", (value))
}

// GetProviderName gets the value of ProviderName for the instance
func (instance *MSFT_WMI_GenericNonCOMEvent) GetPropertyProviderName() (value string, err error) {
	retValue, err := instance.GetProperty("ProviderName")
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
