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

// Win32_ComClassEmulator struct
type Win32_ComClassEmulator struct {
	*cim.WmiInstance

	//
	NewVersion Win32_ClassicCOMClass

	//
	OldVersion Win32_ClassicCOMClass
}

func NewWin32_ComClassEmulatorEx1(instance *cim.WmiInstance) (newInstance *Win32_ComClassEmulator, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_ComClassEmulator{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ComClassEmulatorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ComClassEmulator, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ComClassEmulator{
		WmiInstance: tmp,
	}
	return
}

// SetNewVersion sets the value of NewVersion for the instance
func (instance *Win32_ComClassEmulator) SetPropertyNewVersion(value Win32_ClassicCOMClass) (err error) {
	return instance.SetProperty("NewVersion", (value))
}

// GetNewVersion gets the value of NewVersion for the instance
func (instance *Win32_ComClassEmulator) GetPropertyNewVersion() (value Win32_ClassicCOMClass, err error) {
	retValue, err := instance.GetProperty("NewVersion")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_ClassicCOMClass)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_ClassicCOMClass is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_ClassicCOMClass(valuetmp)

	return
}

// SetOldVersion sets the value of OldVersion for the instance
func (instance *Win32_ComClassEmulator) SetPropertyOldVersion(value Win32_ClassicCOMClass) (err error) {
	return instance.SetProperty("OldVersion", (value))
}

// GetOldVersion gets the value of OldVersion for the instance
func (instance *Win32_ComClassEmulator) GetPropertyOldVersion() (value Win32_ClassicCOMClass, err error) {
	retValue, err := instance.GetProperty("OldVersion")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_ClassicCOMClass)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_ClassicCOMClass is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_ClassicCOMClass(valuetmp)

	return
}
