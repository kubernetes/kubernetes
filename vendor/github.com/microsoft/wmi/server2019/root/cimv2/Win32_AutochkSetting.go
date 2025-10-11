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

// Win32_AutochkSetting struct
type Win32_AutochkSetting struct {
	*CIM_Setting

	//
	UserInputDelay uint32
}

func NewWin32_AutochkSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_AutochkSetting, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_AutochkSetting{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_AutochkSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_AutochkSetting, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_AutochkSetting{
		CIM_Setting: tmp,
	}
	return
}

// SetUserInputDelay sets the value of UserInputDelay for the instance
func (instance *Win32_AutochkSetting) SetPropertyUserInputDelay(value uint32) (err error) {
	return instance.SetProperty("UserInputDelay", (value))
}

// GetUserInputDelay gets the value of UserInputDelay for the instance
func (instance *Win32_AutochkSetting) GetPropertyUserInputDelay() (value uint32, err error) {
	retValue, err := instance.GetProperty("UserInputDelay")
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
