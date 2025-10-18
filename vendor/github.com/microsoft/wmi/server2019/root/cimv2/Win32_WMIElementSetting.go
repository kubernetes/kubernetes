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

// Win32_WMIElementSetting struct
type Win32_WMIElementSetting struct {
	*CIM_ElementSetting
}

func NewWin32_WMIElementSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_WMIElementSetting, err error) {
	tmp, err := NewCIM_ElementSettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_WMIElementSetting{
		CIM_ElementSetting: tmp,
	}
	return
}

func NewWin32_WMIElementSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_WMIElementSetting, err error) {
	tmp, err := NewCIM_ElementSettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_WMIElementSetting{
		CIM_ElementSetting: tmp,
	}
	return
}
