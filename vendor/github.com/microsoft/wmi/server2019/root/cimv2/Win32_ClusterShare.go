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

// Win32_ClusterShare struct
type Win32_ClusterShare struct {
	*Win32_Share

	//
	ServerName string
}

func NewWin32_ClusterShareEx1(instance *cim.WmiInstance) (newInstance *Win32_ClusterShare, err error) {
	tmp, err := NewWin32_ShareEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ClusterShare{
		Win32_Share: tmp,
	}
	return
}

func NewWin32_ClusterShareEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ClusterShare, err error) {
	tmp, err := NewWin32_ShareEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ClusterShare{
		Win32_Share: tmp,
	}
	return
}

// SetServerName sets the value of ServerName for the instance
func (instance *Win32_ClusterShare) SetPropertyServerName(value string) (err error) {
	return instance.SetProperty("ServerName", (value))
}

// GetServerName gets the value of ServerName for the instance
func (instance *Win32_ClusterShare) GetPropertyServerName() (value string, err error) {
	retValue, err := instance.GetProperty("ServerName")
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
