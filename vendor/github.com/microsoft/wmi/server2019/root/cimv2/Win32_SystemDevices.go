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
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Win32_SystemDevices struct
type Win32_SystemDevices struct {
	*CIM_SystemDevice
}

func NewWin32_SystemDevicesEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemDevices, err error) {
	tmp, err := NewCIM_SystemDeviceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemDevices{
		CIM_SystemDevice: tmp,
	}
	return
}

func NewWin32_SystemDevicesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemDevices, err error) {
	tmp, err := NewCIM_SystemDeviceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemDevices{
		CIM_SystemDevice: tmp,
	}
	return
}
