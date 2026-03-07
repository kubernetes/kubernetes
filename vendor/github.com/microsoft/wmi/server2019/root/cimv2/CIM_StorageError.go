// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// CIM_StorageError struct
type CIM_StorageError struct {
	*cim.WmiInstance

	//
	DeviceCreationClassName string

	//
	DeviceID string

	//
	EndingAddress uint64

	//
	StartingAddress uint64

	//
	SystemCreationClassName string

	//
	SystemName string
}

func NewCIM_StorageErrorEx1(instance *cim.WmiInstance) (newInstance *CIM_StorageError, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &CIM_StorageError{
		WmiInstance: tmp,
	}
	return
}

func NewCIM_StorageErrorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_StorageError, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_StorageError{
		WmiInstance: tmp,
	}
	return
}

// SetDeviceCreationClassName sets the value of DeviceCreationClassName for the instance
func (instance *CIM_StorageError) SetPropertyDeviceCreationClassName(value string) (err error) {
	return instance.SetProperty("DeviceCreationClassName", (value))
}

// GetDeviceCreationClassName gets the value of DeviceCreationClassName for the instance
func (instance *CIM_StorageError) GetPropertyDeviceCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceCreationClassName")
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

// SetDeviceID sets the value of DeviceID for the instance
func (instance *CIM_StorageError) SetPropertyDeviceID(value string) (err error) {
	return instance.SetProperty("DeviceID", (value))
}

// GetDeviceID gets the value of DeviceID for the instance
func (instance *CIM_StorageError) GetPropertyDeviceID() (value string, err error) {
	retValue, err := instance.GetProperty("DeviceID")
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

// SetEndingAddress sets the value of EndingAddress for the instance
func (instance *CIM_StorageError) SetPropertyEndingAddress(value uint64) (err error) {
	return instance.SetProperty("EndingAddress", (value))
}

// GetEndingAddress gets the value of EndingAddress for the instance
func (instance *CIM_StorageError) GetPropertyEndingAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("EndingAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetStartingAddress sets the value of StartingAddress for the instance
func (instance *CIM_StorageError) SetPropertyStartingAddress(value uint64) (err error) {
	return instance.SetProperty("StartingAddress", (value))
}

// GetStartingAddress gets the value of StartingAddress for the instance
func (instance *CIM_StorageError) GetPropertyStartingAddress() (value uint64, err error) {
	retValue, err := instance.GetProperty("StartingAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetSystemCreationClassName sets the value of SystemCreationClassName for the instance
func (instance *CIM_StorageError) SetPropertySystemCreationClassName(value string) (err error) {
	return instance.SetProperty("SystemCreationClassName", (value))
}

// GetSystemCreationClassName gets the value of SystemCreationClassName for the instance
func (instance *CIM_StorageError) GetPropertySystemCreationClassName() (value string, err error) {
	retValue, err := instance.GetProperty("SystemCreationClassName")
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

// SetSystemName sets the value of SystemName for the instance
func (instance *CIM_StorageError) SetPropertySystemName(value string) (err error) {
	return instance.SetProperty("SystemName", (value))
}

// GetSystemName gets the value of SystemName for the instance
func (instance *CIM_StorageError) GetPropertySystemName() (value string, err error) {
	retValue, err := instance.GetProperty("SystemName")
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
