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

// Win32_DCOMApplication struct
type Win32_DCOMApplication struct {
	*Win32_COMApplication

	//
	AppID string
}

func NewWin32_DCOMApplicationEx1(instance *cim.WmiInstance) (newInstance *Win32_DCOMApplication, err error) {
	tmp, err := NewWin32_COMApplicationEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_DCOMApplication{
		Win32_COMApplication: tmp,
	}
	return
}

func NewWin32_DCOMApplicationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_DCOMApplication, err error) {
	tmp, err := NewWin32_COMApplicationEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_DCOMApplication{
		Win32_COMApplication: tmp,
	}
	return
}

// SetAppID sets the value of AppID for the instance
func (instance *Win32_DCOMApplication) SetPropertyAppID(value string) (err error) {
	return instance.SetProperty("AppID", (value))
}

// GetAppID gets the value of AppID for the instance
func (instance *Win32_DCOMApplication) GetPropertyAppID() (value string, err error) {
	retValue, err := instance.GetProperty("AppID")
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
