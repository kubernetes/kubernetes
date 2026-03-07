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

// Win32_LogonSessionMappedDisk struct
type Win32_LogonSessionMappedDisk struct {
	*CIM_Dependency
}

func NewWin32_LogonSessionMappedDiskEx1(instance *cim.WmiInstance) (newInstance *Win32_LogonSessionMappedDisk, err error) {
	tmp, err := NewCIM_DependencyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogonSessionMappedDisk{
		CIM_Dependency: tmp,
	}
	return
}

func NewWin32_LogonSessionMappedDiskEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogonSessionMappedDisk, err error) {
	tmp, err := NewCIM_DependencyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogonSessionMappedDisk{
		CIM_Dependency: tmp,
	}
	return
}
