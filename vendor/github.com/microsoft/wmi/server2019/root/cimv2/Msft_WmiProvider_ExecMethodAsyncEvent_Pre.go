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

// Msft_WmiProvider_ExecMethodAsyncEvent_Pre struct
type Msft_WmiProvider_ExecMethodAsyncEvent_Pre struct {
	*Msft_WmiProvider_OperationEvent_Pre

	//
	Flags uint32

	//
	InputParameters interface{}

	//
	MethodName string

	//
	ObjectPath string
}

func NewMsft_WmiProvider_ExecMethodAsyncEvent_PreEx1(instance *cim.WmiInstance) (newInstance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEvent_PreEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_ExecMethodAsyncEvent_Pre{
		Msft_WmiProvider_OperationEvent_Pre: tmp,
	}
	return
}

func NewMsft_WmiProvider_ExecMethodAsyncEvent_PreEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEvent_PreEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_ExecMethodAsyncEvent_Pre{
		Msft_WmiProvider_OperationEvent_Pre: tmp,
	}
	return
}

// SetFlags sets the value of Flags for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) SetPropertyFlags(value uint32) (err error) {
	return instance.SetProperty("Flags", (value))
}

// GetFlags gets the value of Flags for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) GetPropertyFlags() (value uint32, err error) {
	retValue, err := instance.GetProperty("Flags")
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

// SetInputParameters sets the value of InputParameters for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) SetPropertyInputParameters(value interface{}) (err error) {
	return instance.SetProperty("InputParameters", (value))
}

// GetInputParameters gets the value of InputParameters for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) GetPropertyInputParameters() (value interface{}, err error) {
	retValue, err := instance.GetProperty("InputParameters")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(interface{})
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " interface{} is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = interface{}(valuetmp)

	return
}

// SetMethodName sets the value of MethodName for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) SetPropertyMethodName(value string) (err error) {
	return instance.SetProperty("MethodName", (value))
}

// GetMethodName gets the value of MethodName for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) GetPropertyMethodName() (value string, err error) {
	retValue, err := instance.GetProperty("MethodName")
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

// SetObjectPath sets the value of ObjectPath for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) SetPropertyObjectPath(value string) (err error) {
	return instance.SetProperty("ObjectPath", (value))
}

// GetObjectPath gets the value of ObjectPath for the instance
func (instance *Msft_WmiProvider_ExecMethodAsyncEvent_Pre) GetPropertyObjectPath() (value string, err error) {
	retValue, err := instance.GetProperty("ObjectPath")
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
