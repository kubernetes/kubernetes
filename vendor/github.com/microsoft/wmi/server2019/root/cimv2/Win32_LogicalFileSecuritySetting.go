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

// Win32_LogicalFileSecuritySetting struct
type Win32_LogicalFileSecuritySetting struct {
	*Win32_SecuritySetting

	//
	OwnerPermissions bool

	//
	Path string
}

func NewWin32_LogicalFileSecuritySettingEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalFileSecuritySetting, err error) {
	tmp, err := NewWin32_SecuritySettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileSecuritySetting{
		Win32_SecuritySetting: tmp,
	}
	return
}

func NewWin32_LogicalFileSecuritySettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalFileSecuritySetting, err error) {
	tmp, err := NewWin32_SecuritySettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileSecuritySetting{
		Win32_SecuritySetting: tmp,
	}
	return
}

// SetOwnerPermissions sets the value of OwnerPermissions for the instance
func (instance *Win32_LogicalFileSecuritySetting) SetPropertyOwnerPermissions(value bool) (err error) {
	return instance.SetProperty("OwnerPermissions", (value))
}

// GetOwnerPermissions gets the value of OwnerPermissions for the instance
func (instance *Win32_LogicalFileSecuritySetting) GetPropertyOwnerPermissions() (value bool, err error) {
	retValue, err := instance.GetProperty("OwnerPermissions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetPath sets the value of Path for the instance
func (instance *Win32_LogicalFileSecuritySetting) SetPropertyPath(value string) (err error) {
	return instance.SetProperty("Path", (value))
}

// GetPath gets the value of Path for the instance
func (instance *Win32_LogicalFileSecuritySetting) GetPropertyPath() (value string, err error) {
	retValue, err := instance.GetProperty("Path")
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
