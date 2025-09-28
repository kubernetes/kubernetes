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

// Win32_VolumeChangeEvent struct
type Win32_VolumeChangeEvent struct {
	*Win32_DeviceChangeEvent

	//
	DriveName string
}

func NewWin32_VolumeChangeEventEx1(instance *cim.WmiInstance) (newInstance *Win32_VolumeChangeEvent, err error) {
	tmp, err := NewWin32_DeviceChangeEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_VolumeChangeEvent{
		Win32_DeviceChangeEvent: tmp,
	}
	return
}

func NewWin32_VolumeChangeEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_VolumeChangeEvent, err error) {
	tmp, err := NewWin32_DeviceChangeEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_VolumeChangeEvent{
		Win32_DeviceChangeEvent: tmp,
	}
	return
}

// SetDriveName sets the value of DriveName for the instance
func (instance *Win32_VolumeChangeEvent) SetPropertyDriveName(value string) (err error) {
	return instance.SetProperty("DriveName", (value))
}

// GetDriveName gets the value of DriveName for the instance
func (instance *Win32_VolumeChangeEvent) GetPropertyDriveName() (value string, err error) {
	retValue, err := instance.GetProperty("DriveName")
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
