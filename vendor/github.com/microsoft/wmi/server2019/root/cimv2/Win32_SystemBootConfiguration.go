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

// Win32_SystemBootConfiguration struct
type Win32_SystemBootConfiguration struct {
	*Win32_SystemSetting
}

func NewWin32_SystemBootConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemBootConfiguration, err error) {
	tmp, err := NewWin32_SystemSettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemBootConfiguration{
		Win32_SystemSetting: tmp,
	}
	return
}

func NewWin32_SystemBootConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemBootConfiguration, err error) {
	tmp, err := NewWin32_SystemSettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemBootConfiguration{
		Win32_SystemSetting: tmp,
	}
	return
}
