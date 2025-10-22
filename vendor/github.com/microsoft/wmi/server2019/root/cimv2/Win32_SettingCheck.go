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

// Win32_SettingCheck struct
type Win32_SettingCheck struct {
	*cim.WmiInstance

	//
	Check CIM_Check

	//
	Setting CIM_Setting
}

func NewWin32_SettingCheckEx1(instance *cim.WmiInstance) (newInstance *Win32_SettingCheck, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_SettingCheck{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_SettingCheckEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SettingCheck, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SettingCheck{
		WmiInstance: tmp,
	}
	return
}

// SetCheck sets the value of Check for the instance
func (instance *Win32_SettingCheck) SetPropertyCheck(value CIM_Check) (err error) {
	return instance.SetProperty("Check", (value))
}

// GetCheck gets the value of Check for the instance
func (instance *Win32_SettingCheck) GetPropertyCheck() (value CIM_Check, err error) {
	retValue, err := instance.GetProperty("Check")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Check)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Check is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Check(valuetmp)

	return
}

// SetSetting sets the value of Setting for the instance
func (instance *Win32_SettingCheck) SetPropertySetting(value CIM_Setting) (err error) {
	return instance.SetProperty("Setting", (value))
}

// GetSetting gets the value of Setting for the instance
func (instance *Win32_SettingCheck) GetPropertySetting() (value CIM_Setting, err error) {
	retValue, err := instance.GetProperty("Setting")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(CIM_Setting)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " CIM_Setting is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = CIM_Setting(valuetmp)

	return
}
