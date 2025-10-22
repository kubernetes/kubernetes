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

// Win32_SystemConfigurationChangeEvent struct
type Win32_SystemConfigurationChangeEvent struct {
	*Win32_DeviceChangeEvent
}

func NewWin32_SystemConfigurationChangeEventEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemConfigurationChangeEvent, err error) {
	tmp, err := NewWin32_DeviceChangeEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemConfigurationChangeEvent{
		Win32_DeviceChangeEvent: tmp,
	}
	return
}

func NewWin32_SystemConfigurationChangeEventEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemConfigurationChangeEvent, err error) {
	tmp, err := NewWin32_DeviceChangeEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemConfigurationChangeEvent{
		Win32_DeviceChangeEvent: tmp,
	}
	return
}
