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

// Win32_SecuritySettingOfLogicalShare struct
type Win32_SecuritySettingOfLogicalShare struct {
	*Win32_SecuritySettingOfObject
}

func NewWin32_SecuritySettingOfLogicalShareEx1(instance *cim.WmiInstance) (newInstance *Win32_SecuritySettingOfLogicalShare, err error) {
	tmp, err := NewWin32_SecuritySettingOfObjectEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SecuritySettingOfLogicalShare{
		Win32_SecuritySettingOfObject: tmp,
	}
	return
}

func NewWin32_SecuritySettingOfLogicalShareEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SecuritySettingOfLogicalShare, err error) {
	tmp, err := NewWin32_SecuritySettingOfObjectEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SecuritySettingOfLogicalShare{
		Win32_SecuritySettingOfObject: tmp,
	}
	return
}
