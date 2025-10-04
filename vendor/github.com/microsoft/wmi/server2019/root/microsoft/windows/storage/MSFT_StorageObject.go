// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.Microsoft.Windows.Storage
//////////////////////////////////////////////
package storage

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// MSFT_StorageObject struct
type MSFT_StorageObject struct {
	*cim.WmiInstance

	//
	ObjectId string

	//
	PassThroughClass string

	//
	PassThroughIds string

	//
	PassThroughNamespace string

	//
	PassThroughServer string

	//
	UniqueId string
}

func NewMSFT_StorageObjectEx1(instance *cim.WmiInstance) (newInstance *MSFT_StorageObject, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageObject{
		WmiInstance: tmp,
	}
	return
}

func NewMSFT_StorageObjectEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *MSFT_StorageObject, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &MSFT_StorageObject{
		WmiInstance: tmp,
	}
	return
}

// SetObjectId sets the value of ObjectId for the instance
func (instance *MSFT_StorageObject) SetPropertyObjectId(value string) (err error) {
	return instance.SetProperty("ObjectId", (value))
}

// GetObjectId gets the value of ObjectId for the instance
func (instance *MSFT_StorageObject) GetPropertyObjectId() (value string, err error) {
	retValue, err := instance.GetProperty("ObjectId")
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

// SetPassThroughClass sets the value of PassThroughClass for the instance
func (instance *MSFT_StorageObject) SetPropertyPassThroughClass(value string) (err error) {
	return instance.SetProperty("PassThroughClass", (value))
}

// GetPassThroughClass gets the value of PassThroughClass for the instance
func (instance *MSFT_StorageObject) GetPropertyPassThroughClass() (value string, err error) {
	retValue, err := instance.GetProperty("PassThroughClass")
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

// SetPassThroughIds sets the value of PassThroughIds for the instance
func (instance *MSFT_StorageObject) SetPropertyPassThroughIds(value string) (err error) {
	return instance.SetProperty("PassThroughIds", (value))
}

// GetPassThroughIds gets the value of PassThroughIds for the instance
func (instance *MSFT_StorageObject) GetPropertyPassThroughIds() (value string, err error) {
	retValue, err := instance.GetProperty("PassThroughIds")
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

// SetPassThroughNamespace sets the value of PassThroughNamespace for the instance
func (instance *MSFT_StorageObject) SetPropertyPassThroughNamespace(value string) (err error) {
	return instance.SetProperty("PassThroughNamespace", (value))
}

// GetPassThroughNamespace gets the value of PassThroughNamespace for the instance
func (instance *MSFT_StorageObject) GetPropertyPassThroughNamespace() (value string, err error) {
	retValue, err := instance.GetProperty("PassThroughNamespace")
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

// SetPassThroughServer sets the value of PassThroughServer for the instance
func (instance *MSFT_StorageObject) SetPropertyPassThroughServer(value string) (err error) {
	return instance.SetProperty("PassThroughServer", (value))
}

// GetPassThroughServer gets the value of PassThroughServer for the instance
func (instance *MSFT_StorageObject) GetPropertyPassThroughServer() (value string, err error) {
	retValue, err := instance.GetProperty("PassThroughServer")
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

// SetUniqueId sets the value of UniqueId for the instance
func (instance *MSFT_StorageObject) SetPropertyUniqueId(value string) (err error) {
	return instance.SetProperty("UniqueId", (value))
}

// GetUniqueId gets the value of UniqueId for the instance
func (instance *MSFT_StorageObject) GetPropertyUniqueId() (value string, err error) {
	retValue, err := instance.GetProperty("UniqueId")
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
