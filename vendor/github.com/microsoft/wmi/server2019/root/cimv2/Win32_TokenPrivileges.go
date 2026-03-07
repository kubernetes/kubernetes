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

// Win32_TokenPrivileges struct
type Win32_TokenPrivileges struct {
	*cim.WmiInstance

	//
	PrivilegeCount uint32

	//
	Privileges []Win32_LUIDandAttributes
}

func NewWin32_TokenPrivilegesEx1(instance *cim.WmiInstance) (newInstance *Win32_TokenPrivileges, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_TokenPrivileges{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_TokenPrivilegesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_TokenPrivileges, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_TokenPrivileges{
		WmiInstance: tmp,
	}
	return
}

// SetPrivilegeCount sets the value of PrivilegeCount for the instance
func (instance *Win32_TokenPrivileges) SetPropertyPrivilegeCount(value uint32) (err error) {
	return instance.SetProperty("PrivilegeCount", (value))
}

// GetPrivilegeCount gets the value of PrivilegeCount for the instance
func (instance *Win32_TokenPrivileges) GetPropertyPrivilegeCount() (value uint32, err error) {
	retValue, err := instance.GetProperty("PrivilegeCount")
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

// SetPrivileges sets the value of Privileges for the instance
func (instance *Win32_TokenPrivileges) SetPropertyPrivileges(value []Win32_LUIDandAttributes) (err error) {
	return instance.SetProperty("Privileges", (value))
}

// GetPrivileges gets the value of Privileges for the instance
func (instance *Win32_TokenPrivileges) GetPropertyPrivileges() (value []Win32_LUIDandAttributes, err error) {
	retValue, err := instance.GetProperty("Privileges")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(Win32_LUIDandAttributes)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " Win32_LUIDandAttributes is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, Win32_LUIDandAttributes(valuetmp))
	}

	return
}
