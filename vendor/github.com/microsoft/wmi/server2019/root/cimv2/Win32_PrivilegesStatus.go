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

// Win32_PrivilegesStatus struct
type Win32_PrivilegesStatus struct {
	*__ExtendedStatus

	//
	PrivilegesNotHeld []string

	//
	PrivilegesRequired []string
}

func NewWin32_PrivilegesStatusEx1(instance *cim.WmiInstance) (newInstance *Win32_PrivilegesStatus, err error) {
	tmp, err := New__ExtendedStatusEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PrivilegesStatus{
		__ExtendedStatus: tmp,
	}
	return
}

func NewWin32_PrivilegesStatusEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PrivilegesStatus, err error) {
	tmp, err := New__ExtendedStatusEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PrivilegesStatus{
		__ExtendedStatus: tmp,
	}
	return
}

// SetPrivilegesNotHeld sets the value of PrivilegesNotHeld for the instance
func (instance *Win32_PrivilegesStatus) SetPropertyPrivilegesNotHeld(value []string) (err error) {
	return instance.SetProperty("PrivilegesNotHeld", (value))
}

// GetPrivilegesNotHeld gets the value of PrivilegesNotHeld for the instance
func (instance *Win32_PrivilegesStatus) GetPropertyPrivilegesNotHeld() (value []string, err error) {
	retValue, err := instance.GetProperty("PrivilegesNotHeld")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetPrivilegesRequired sets the value of PrivilegesRequired for the instance
func (instance *Win32_PrivilegesStatus) SetPropertyPrivilegesRequired(value []string) (err error) {
	return instance.SetProperty("PrivilegesRequired", (value))
}

// GetPrivilegesRequired gets the value of PrivilegesRequired for the instance
func (instance *Win32_PrivilegesStatus) GetPropertyPrivilegesRequired() (value []string, err error) {
	retValue, err := instance.GetProperty("PrivilegesRequired")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}
