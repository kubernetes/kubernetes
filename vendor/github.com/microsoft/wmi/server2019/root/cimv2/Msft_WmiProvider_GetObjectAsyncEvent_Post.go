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

// Msft_WmiProvider_GetObjectAsyncEvent_Post struct
type Msft_WmiProvider_GetObjectAsyncEvent_Post struct {
	*Msft_WmiProvider_OperationEvent_Post

	//
	Flags uint32

	//
	ObjectParameter interface{}

	//
	ObjectPath string

	//
	ResultCode uint32

	//
	StringParameter string
}

func NewMsft_WmiProvider_GetObjectAsyncEvent_PostEx1(instance *cim.WmiInstance) (newInstance *Msft_WmiProvider_GetObjectAsyncEvent_Post, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEvent_PostEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_GetObjectAsyncEvent_Post{
		Msft_WmiProvider_OperationEvent_Post: tmp,
	}
	return
}

func NewMsft_WmiProvider_GetObjectAsyncEvent_PostEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Msft_WmiProvider_GetObjectAsyncEvent_Post, err error) {
	tmp, err := NewMsft_WmiProvider_OperationEvent_PostEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Msft_WmiProvider_GetObjectAsyncEvent_Post{
		Msft_WmiProvider_OperationEvent_Post: tmp,
	}
	return
}

// SetFlags sets the value of Flags for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) SetPropertyFlags(value uint32) (err error) {
	return instance.SetProperty("Flags", (value))
}

// GetFlags gets the value of Flags for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) GetPropertyFlags() (value uint32, err error) {
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

// SetObjectParameter sets the value of ObjectParameter for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) SetPropertyObjectParameter(value interface{}) (err error) {
	return instance.SetProperty("ObjectParameter", (value))
}

// GetObjectParameter gets the value of ObjectParameter for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) GetPropertyObjectParameter() (value interface{}, err error) {
	retValue, err := instance.GetProperty("ObjectParameter")
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

// SetObjectPath sets the value of ObjectPath for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) SetPropertyObjectPath(value string) (err error) {
	return instance.SetProperty("ObjectPath", (value))
}

// GetObjectPath gets the value of ObjectPath for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) GetPropertyObjectPath() (value string, err error) {
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

// SetResultCode sets the value of ResultCode for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) SetPropertyResultCode(value uint32) (err error) {
	return instance.SetProperty("ResultCode", (value))
}

// GetResultCode gets the value of ResultCode for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) GetPropertyResultCode() (value uint32, err error) {
	retValue, err := instance.GetProperty("ResultCode")
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

// SetStringParameter sets the value of StringParameter for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) SetPropertyStringParameter(value string) (err error) {
	return instance.SetProperty("StringParameter", (value))
}

// GetStringParameter gets the value of StringParameter for the instance
func (instance *Msft_WmiProvider_GetObjectAsyncEvent_Post) GetPropertyStringParameter() (value string, err error) {
	retValue, err := instance.GetProperty("StringParameter")
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
