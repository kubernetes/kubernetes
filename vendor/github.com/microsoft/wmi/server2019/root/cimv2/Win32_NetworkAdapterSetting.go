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

// Win32_NetworkAdapterSetting struct
type Win32_NetworkAdapterSetting struct {
	*Win32_DeviceSettings
}

func NewWin32_NetworkAdapterSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_NetworkAdapterSetting, err error) {
	tmp, err := NewWin32_DeviceSettingsEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkAdapterSetting{
		Win32_DeviceSettings: tmp,
	}
	return
}

func NewWin32_NetworkAdapterSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NetworkAdapterSetting, err error) {
	tmp, err := NewWin32_DeviceSettingsEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkAdapterSetting{
		Win32_DeviceSettings: tmp,
	}
	return
}
