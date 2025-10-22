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

// Win32_LogicalFileGroup struct
type Win32_LogicalFileGroup struct {
	*Win32_SecuritySettingGroup
}

func NewWin32_LogicalFileGroupEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalFileGroup, err error) {
	tmp, err := NewWin32_SecuritySettingGroupEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileGroup{
		Win32_SecuritySettingGroup: tmp,
	}
	return
}

func NewWin32_LogicalFileGroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalFileGroup, err error) {
	tmp, err := NewWin32_SecuritySettingGroupEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileGroup{
		Win32_SecuritySettingGroup: tmp,
	}
	return
}
