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

// Win32_ClassicCOMClassSettings struct
type Win32_ClassicCOMClassSettings struct {
	*CIM_ElementSetting
}

func NewWin32_ClassicCOMClassSettingsEx1(instance *cim.WmiInstance) (newInstance *Win32_ClassicCOMClassSettings, err error) {
	tmp, err := NewCIM_ElementSettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassicCOMClassSettings{
		CIM_ElementSetting: tmp,
	}
	return
}

func NewWin32_ClassicCOMClassSettingsEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ClassicCOMClassSettings, err error) {
	tmp, err := NewCIM_ElementSettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassicCOMClassSettings{
		CIM_ElementSetting: tmp,
	}
	return
}
