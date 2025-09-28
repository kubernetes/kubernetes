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

// Win32_VolumeQuotaSetting struct
type Win32_VolumeQuotaSetting struct {
	*CIM_ElementSetting
}

func NewWin32_VolumeQuotaSettingEx1(instance *cim.WmiInstance) (newInstance *Win32_VolumeQuotaSetting, err error) {
	tmp, err := NewCIM_ElementSettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_VolumeQuotaSetting{
		CIM_ElementSetting: tmp,
	}
	return
}

func NewWin32_VolumeQuotaSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_VolumeQuotaSetting, err error) {
	tmp, err := NewCIM_ElementSettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_VolumeQuotaSetting{
		CIM_ElementSetting: tmp,
	}
	return
}
