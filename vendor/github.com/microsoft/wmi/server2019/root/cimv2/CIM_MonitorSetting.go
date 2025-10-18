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

// CIM_MonitorSetting struct
type CIM_MonitorSetting struct {
	*CIM_ElementSetting
}

func NewCIM_MonitorSettingEx1(instance *cim.WmiInstance) (newInstance *CIM_MonitorSetting, err error) {
	tmp, err := NewCIM_ElementSettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_MonitorSetting{
		CIM_ElementSetting: tmp,
	}
	return
}

func NewCIM_MonitorSettingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_MonitorSetting, err error) {
	tmp, err := NewCIM_ElementSettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_MonitorSetting{
		CIM_ElementSetting: tmp,
	}
	return
}
