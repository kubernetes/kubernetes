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

// Win32_LogicalShareAuditing struct
type Win32_LogicalShareAuditing struct {
	*Win32_SecuritySettingAuditing
}

func NewWin32_LogicalShareAuditingEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalShareAuditing, err error) {
	tmp, err := NewWin32_SecuritySettingAuditingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalShareAuditing{
		Win32_SecuritySettingAuditing: tmp,
	}
	return
}

func NewWin32_LogicalShareAuditingEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalShareAuditing, err error) {
	tmp, err := NewWin32_SecuritySettingAuditingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalShareAuditing{
		Win32_SecuritySettingAuditing: tmp,
	}
	return
}
