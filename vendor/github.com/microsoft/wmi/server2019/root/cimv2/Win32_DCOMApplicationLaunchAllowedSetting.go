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

// Win32_DCOMApplicationLaunchAllowedSetting struct
type Win32_DCOMApplicationLaunchAllowedSetting struct {
	*cim.WmiInstance

	//
	Element Win32_DCOMApplication

	//
	Setting Win32_SID
}

func NewWin32_DCOMApplicationLaunchAllowedSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_DCOMApplicationLaunchAllowedSetting, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_DCOMApplicationLaunchAllowedSetting{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_DCOMApplicationLaunchAllowedSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DCOMApplicationLaunchAllowedSetting, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DCOMApplicationLaunchAllowedSetting{
		WmiInstance: tmp,
	}
	return
}

// SetElement sets the value of Element for the instance
func (instance *Win32_DCOMApplicationLaunchAllowedSetting) SetPropertyElement(value Win32_DCOMApplication) (err error) {
	return instance.SetProperty("Element", (value))
}

// GetElement gets the value of Element for the instance
func (instance *Win32_DCOMApplicationLaunchAllowedSetting) GetPropertyElement() (value Win32_DCOMApplication, err error) {
	retValue, err := instance.GetProperty("Element")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_DCOMApplication)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_DCOMApplication is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_DCOMApplication(valuetmp)

	return
}

// SetSetting sets the value of Setting for the instance
func (instance *Win32_DCOMApplicationLaunchAllowedSetting) SetPropertySetting(value Win32_SID) (err error) {
	return instance.SetProperty("Setting", (value))
}

// GetSetting gets the value of Setting for the instance
func (instance *Win32_DCOMApplicationLaunchAllowedSetting) GetPropertySetting() (value Win32_SID, err error) {
	retValue, err := instance.GetProperty("Setting")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(Win32_SID)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " Win32_SID is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Win32_SID(valuetmp)

	return
}
