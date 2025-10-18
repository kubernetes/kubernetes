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

// Msft_WmiProvider_OperationEvent struct
type Msft_WmiProvider_OperationEvent struct {
	*MSFT_WmiSelfEvent

	//
	HostingGroup string

	//
	HostingSpecification uint32

	//
	Locale string

	//
	Namespace string

	//
	provider string

	//
	TransactionIdentifer string

	//
	User string
}

func NewMsft_WmiProvider_OperationEventEx1(instance *cim.WmiInstance) (newInstance *Msft_WmiProvider_OperationEvent, err error) {
	tmp, err := NewMSFT_WmiSelfEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_OperationEvent{
		MSFT_WmiSelfEvent: tmp,
	}
	return
}

func NewMsft_WmiProvider_OperationEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Msft_WmiProvider_OperationEvent, err error) {
	tmp, err := NewMSFT_WmiSelfEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_OperationEvent{
		MSFT_WmiSelfEvent: tmp,
	}
	return
}

// SetHostingGroup sets the value of HostingGroup for the instance
func (instance *Msft_WmiProvider_OperationEvent) SetPropertyHostingGroup(value string) (err error) {
	return instance.SetProperty("HostingGroup", (value))
}

// GetHostingGroup gets the value of HostingGroup for the instance
func (instance *Msft_WmiProvider_OperationEvent) GetPropertyHostingGroup() (value string, err error) {
	retValue, err := instance.GetProperty("HostingGroup")
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

// SetHostingSpecification sets the value of HostingSpecification for the instance
func (instance *Msft_WmiProvider_OperationEvent) SetPropertyHostingSpecification(value uint32) (err error) {
	return instance.SetProperty("HostingSpecification", (value))
}

// GetHostingSpecification gets the value of HostingSpecification for the instance
func (instance *Msft_WmiProvider_OperationEvent) GetPropertyHostingSpecification() (value uint32, err error) {
	retValue, err := instance.GetProperty("HostingSpecification")
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

// SetLocale sets the value of Locale for the instance
func (instance *Msft_WmiProvider_OperationEvent) SetPropertyLocale(value string) (err error) {
	return instance.SetProperty("Locale", (value))
}

// GetLocale gets the value of Locale for the instance
func (instance *Msft_WmiProvider_OperationEvent) GetPropertyLocale() (value string, err error) {
	retValue, err := instance.GetProperty("Locale")
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

// SetNamespace sets the value of Namespace for the instance
func (instance *Msft_WmiProvider_OperationEvent) SetPropertyNamespace(value string) (err error) {
	return instance.SetProperty("Namespace", (value))
}

// GetNamespace gets the value of Namespace for the instance
func (instance *Msft_WmiProvider_OperationEvent) GetPropertyNamespace() (value string, err error) {
	retValue, err := instance.GetProperty("Namespace")
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

// Setprovider sets the value of provider for the instance
func (instance *Msft_WmiProvider_OperationEvent) SetPropertyprovider(value string) (err error) {
	return instance.SetProperty("provider", (value))
}

// Getprovider gets the value of provider for the instance
func (instance *Msft_WmiProvider_OperationEvent) GetPropertyprovider() (value string, err error) {
	retValue, err := instance.GetProperty("provider")
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

// SetTransactionIdentifer sets the value of TransactionIdentifer for the instance
func (instance *Msft_WmiProvider_OperationEvent) SetPropertyTransactionIdentifer(value string) (err error) {
	return instance.SetProperty("TransactionIdentifer", (value))
}

// GetTransactionIdentifer gets the value of TransactionIdentifer for the instance
func (instance *Msft_WmiProvider_OperationEvent) GetPropertyTransactionIdentifer() (value string, err error) {
	retValue, err := instance.GetProperty("TransactionIdentifer")
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

// SetUser sets the value of User for the instance
func (instance *Msft_WmiProvider_OperationEvent) SetPropertyUser(value string) (err error) {
	return instance.SetProperty("User", (value))
}

// GetUser gets the value of User for the instance
func (instance *Msft_WmiProvider_OperationEvent) GetPropertyUser() (value string, err error) {
	retValue, err := instance.GetProperty("User")
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
