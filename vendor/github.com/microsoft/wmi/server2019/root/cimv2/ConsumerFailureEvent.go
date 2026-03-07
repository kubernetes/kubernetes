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

// __ConsumerFailureEvent struct
type __ConsumerFailureEvent struct {
	*__EventDroppedEvent

	//
	ErrorCode uint32

	//
	ErrorDescription string

	//
	ErrorObject __ExtendedStatus
}

func New__ConsumerFailureEventEx1(instance *cim.WmiInstance) (newInstance *__ConsumerFailureEvent, err error) {
	tmp, err := New__EventDroppedEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &__ConsumerFailureEvent{
		__EventDroppedEvent: tmp,
	}
	return
}

func New__ConsumerFailureEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *__ConsumerFailureEvent, err error) {
	tmp, err := New__EventDroppedEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &__ConsumerFailureEvent{
		__EventDroppedEvent: tmp,
	}
	return
}

// SetErrorCode sets the value of ErrorCode for the instance
func (instance *__ConsumerFailureEvent) SetPropertyErrorCode(value uint32) (err error) {
	return instance.SetProperty("ErrorCode", (value))
}

// GetErrorCode gets the value of ErrorCode for the instance
func (instance *__ConsumerFailureEvent) GetPropertyErrorCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("ErrorCode")
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

// SetErrorDescription sets the value of ErrorDescription for the instance
func (instance *__ConsumerFailureEvent) SetPropertyErrorDescription(value string) (err error) {
	return instance.SetProperty("ErrorDescription", (value))
}

// GetErrorDescription gets the value of ErrorDescription for the instance
func (instance *__ConsumerFailureEvent) GetPropertyErrorDescription() (value string, err error) {
	retValue, err := instance.GetProperty("ErrorDescription")
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

// SetErrorObject sets the value of ErrorObject for the instance
func (instance *__ConsumerFailureEvent) SetPropertyErrorObject(value __ExtendedStatus) (err error) {
	return instance.SetProperty("ErrorObject", (value))
}

// GetErrorObject gets the value of ErrorObject for the instance
func (instance *__ConsumerFailureEvent) GetPropertyErrorObject() (value __ExtendedStatus, err error) {
	retValue, err := instance.GetProperty("ErrorObject")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(__ExtendedStatus)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " __ExtendedStatus is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = __ExtendedStatus(valuetmp)

	return
}
