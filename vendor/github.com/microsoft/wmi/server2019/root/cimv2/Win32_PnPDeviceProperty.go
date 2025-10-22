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

// Win32_PnPDeviceProperty struct
type Win32_PnPDeviceProperty struct {
	*cim.WmiInstance

	//
	DeviceID string

	//
	key string

	//
	KeyName string

	//
	Type uint32
}

func NewWin32_PnPDevicePropertyEx1(instance *cim.WmiInstance) (newInstance *Win32_PnPDeviceProperty, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_PnPDeviceProperty{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_PnPDevicePropertyEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PnPDeviceProperty, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PnPDeviceProperty{
		WmiInstance: tmp,
	}
	return
}

// SetDeviceID sets the value of DeviceID for the instance
func (instance *Win32_PnPDeviceProperty) SetPropertyDeviceID(value string) (err error) {
	return instance.SetProperty("DeviceID", (value))
}

// GetDeviceID gets the value of DeviceID for the instance
func (instance *Win32_PnPDeviceProperty) GetPropertyDeviceID() (value string, err error) {
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

// Setkey sets the value of key for the instance
func (instance *Win32_PnPDeviceProperty) SetPropertykey(value string) (err error) {
	return instance.SetProperty("key", (value))
}

// Getkey gets the value of key for the instance
func (instance *Win32_PnPDeviceProperty) GetPropertykey() (value string, err error) {
	retValue, err := instance.GetProperty("key")
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

// SetKeyName sets the value of KeyName for the instance
func (instance *Win32_PnPDeviceProperty) SetPropertyKeyName(value string) (err error) {
	return instance.SetProperty("KeyName", (value))
}

// GetKeyName gets the value of KeyName for the instance
func (instance *Win32_PnPDeviceProperty) GetPropertyKeyName() (value string, err error) {
	retValue, err := instance.GetProperty("KeyName")
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

// SetType sets the value of Type for the instance
func (instance *Win32_PnPDeviceProperty) SetPropertyType(value uint32) (err error) {
	return instance.SetProperty("Type", (value))
}

// GetType gets the value of Type for the instance
func (instance *Win32_PnPDeviceProperty) GetPropertyType() (value uint32, err error) {
	retValue, err := instance.GetProperty("Type")
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
