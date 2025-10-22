// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// __MethodInvocationEvent struct
type __MethodInvocationEvent struct {
	*__InstanceOperationEvent

	//
	Method string

	//
	Parameters interface{}

	//
	PreCall bool
}

func New__MethodInvocationEventEx1(instance *cim.WmiInstance) (newInstance *__MethodInvocationEvent, err error) {
	tmp, err := New__InstanceOperationEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__MethodInvocationEvent{
		__InstanceOperationEvent: tmp,
	}
	return
}

func New__MethodInvocationEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__MethodInvocationEvent, err error) {
	tmp, err := New__InstanceOperationEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__MethodInvocationEvent{
		__InstanceOperationEvent: tmp,
	}
	return
}

// SetMethod sets the value of Method for the instance
func (instance *__MethodInvocationEvent) SetPropertyMethod(value string) (err error) {
	return instance.SetProperty("Method", (value))
}

// GetMethod gets the value of Method for the instance
func (instance *__MethodInvocationEvent) GetPropertyMethod() (value string, err error) {
	retValue, err := instance.GetProperty("Method")
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

// SetParameters sets the value of Parameters for the instance
func (instance *__MethodInvocationEvent) SetPropertyParameters(value interface{}) (err error) {
	return instance.SetProperty("Parameters", (value))
}

// GetParameters gets the value of Parameters for the instance
func (instance *__MethodInvocationEvent) GetPropertyParameters() (value interface{}, err error) {
	retValue, err := instance.GetProperty("Parameters")
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

// SetPreCall sets the value of PreCall for the instance
func (instance *__MethodInvocationEvent) SetPropertyPreCall(value bool) (err error) {
	return instance.SetProperty("PreCall", (value))
}

// GetPreCall gets the value of PreCall for the instance
func (instance *__MethodInvocationEvent) GetPropertyPreCall() (value bool, err error) {
	retValue, err := instance.GetProperty("PreCall")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}
