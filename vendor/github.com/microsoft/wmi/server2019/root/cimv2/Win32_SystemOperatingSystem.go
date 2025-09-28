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

// Win32_SystemOperatingSystem struct
type Win32_SystemOperatingSystem struct {
	*CIM_InstalledOS
}

func NewWin32_SystemOperatingSystemEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemOperatingSystem, err error) {
	tmp, err := NewCIM_InstalledOSEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemOperatingSystem{
		CIM_InstalledOS: tmp,
	}
	return
}

func NewWin32_SystemOperatingSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemOperatingSystem, err error) {
	tmp, err := NewCIM_InstalledOSEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemOperatingSystem{
		CIM_InstalledOS: tmp,
	}
	return
}
