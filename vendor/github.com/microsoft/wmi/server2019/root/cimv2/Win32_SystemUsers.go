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

// Win32_SystemUsers struct
type Win32_SystemUsers struct {
	*CIM_SystemComponent
}

func NewWin32_SystemUsersEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemUsers, err error) {
	tmp, err := NewCIM_SystemComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemUsers{
		CIM_SystemComponent: tmp,
	}
	return
}

func NewWin32_SystemUsersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemUsers, err error) {
	tmp, err := NewCIM_SystemComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemUsers{
		CIM_SystemComponent: tmp,
	}
	return
}
