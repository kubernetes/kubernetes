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

// Win32_LogicalFileAccess struct
type Win32_LogicalFileAccess struct {
	*Win32_SecuritySettingAccess
}

func NewWin32_LogicalFileAccessEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalFileAccess, err error) {
	tmp, err := NewWin32_SecuritySettingAccessEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileAccess{
		Win32_SecuritySettingAccess: tmp,
	}
	return
}

func NewWin32_LogicalFileAccessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalFileAccess, err error) {
	tmp, err := NewWin32_SecuritySettingAccessEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileAccess{
		Win32_SecuritySettingAccess: tmp,
	}
	return
}
