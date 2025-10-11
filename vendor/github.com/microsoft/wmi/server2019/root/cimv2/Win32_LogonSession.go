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

// Win32_LogonSession struct
type Win32_LogonSession struct {
	*Win32_Session

	//
	AuthenticationPackage string

	//
	LogonId string

	//
	LogonType uint32
}

func NewWin32_LogonSessionEx1(instance *cim.WmiInstance) (newInstance *Win32_LogonSession, err error) {
	tmp, err := NewWin32_SessionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogonSession{
		Win32_Session: tmp,
	}
	return
}

func NewWin32_LogonSessionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogonSession, err error) {
	tmp, err := NewWin32_SessionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogonSession{
		Win32_Session: tmp,
	}
	return
}

// SetAuthenticationPackage sets the value of AuthenticationPackage for the instance
func (instance *Win32_LogonSession) SetPropertyAuthenticationPackage(value string) (err error) {
	return instance.SetProperty("AuthenticationPackage", (value))
}

// GetAuthenticationPackage gets the value of AuthenticationPackage for the instance
func (instance *Win32_LogonSession) GetPropertyAuthenticationPackage() (value string, err error) {
	retValue, err := instance.GetProperty("AuthenticationPackage")
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

// SetLogonId sets the value of LogonId for the instance
func (instance *Win32_LogonSession) SetPropertyLogonId(value string) (err error) {
	return instance.SetProperty("LogonId", (value))
}

// GetLogonId gets the value of LogonId for the instance
func (instance *Win32_LogonSession) GetPropertyLogonId() (value string, err error) {
	retValue, err := instance.GetProperty("LogonId")
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

// SetLogonType sets the value of LogonType for the instance
func (instance *Win32_LogonSession) SetPropertyLogonType(value uint32) (err error) {
	return instance.SetProperty("LogonType", (value))
}

// GetLogonType gets the value of LogonType for the instance
func (instance *Win32_LogonSession) GetPropertyLogonType() (value uint32, err error) {
	retValue, err := instance.GetProperty("LogonType")
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
