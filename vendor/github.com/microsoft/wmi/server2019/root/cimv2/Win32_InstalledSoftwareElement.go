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

// Win32_InstalledSoftwareElement struct
type Win32_InstalledSoftwareElement struct {
	*CIM_InstalledSoftwareElement
}

func NewWin32_InstalledSoftwareElementEx1(instance *cim.WmiInstance) (newInstance *Win32_InstalledSoftwareElement, err error) {
	tmp, err := NewCIM_InstalledSoftwareElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_InstalledSoftwareElement{
		CIM_InstalledSoftwareElement: tmp,
	}
	return
}

func NewWin32_InstalledSoftwareElementEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_InstalledSoftwareElement, err error) {
	tmp, err := NewCIM_InstalledSoftwareElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_InstalledSoftwareElement{
		CIM_InstalledSoftwareElement: tmp,
	}
	return
}
