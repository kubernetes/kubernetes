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

// Win32_MountPoint struct
type Win32_MountPoint struct {
	*cim.WmiInstance

	//
	Directory Win32_Directory

	//
	Volume Win32_Volume
}

func NewWin32_MountPointEx1(instance *cim.WmiInstance) (newInstance *Win32_MountPoint, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_MountPoint{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_MountPointEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_MountPoint, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_MountPoint{
		WmiInstance: tmp,
	}
	return
}

// SetDirectory sets the value of Directory for the instance
func (instance *Win32_MountPoint) SetPropertyDirectory(value Win32_Directory) (err error) {
	return instance.SetProperty("Directory", (value))
}

// GetDirectory gets the value of Directory for the instance
func (instance *Win32_MountPoint) GetPropertyDirectory() (value Win32_Directory, err error) {
	retValue, err := instance.GetProperty("Directory")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_Directory)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_Directory is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_Directory(valuetmp)

	return
}

// SetVolume sets the value of Volume for the instance
func (instance *Win32_MountPoint) SetPropertyVolume(value Win32_Volume) (err error) {
	return instance.SetProperty("Volume", (value))
}

// GetVolume gets the value of Volume for the instance
func (instance *Win32_MountPoint) GetPropertyVolume() (value Win32_Volume, err error) {
	retValue, err := instance.GetProperty("Volume")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_Volume)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_Volume is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_Volume(valuetmp)

	return
}
