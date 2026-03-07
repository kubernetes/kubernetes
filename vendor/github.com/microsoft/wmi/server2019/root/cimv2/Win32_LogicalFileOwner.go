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

// Win32_LogicalFileOwner struct
type Win32_LogicalFileOwner struct {
	*Win32_SecuritySettingOwner
}

func NewWin32_LogicalFileOwnerEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalFileOwner, err error) {
	tmp, err := NewWin32_SecuritySettingOwnerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileOwner{
		Win32_SecuritySettingOwner: tmp,
	}
	return
}

func NewWin32_LogicalFileOwnerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalFileOwner, err error) {
	tmp, err := NewWin32_SecuritySettingOwnerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalFileOwner{
		Win32_SecuritySettingOwner: tmp,
	}
	return
}
