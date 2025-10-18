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

// Win32_SystemResources struct
type Win32_SystemResources struct {
	*CIM_ComputerSystemResource
}

func NewWin32_SystemResourcesEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemResources, err error) {
	tmp, err := NewCIM_ComputerSystemResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemResources{
		CIM_ComputerSystemResource: tmp,
	}
	return
}

func NewWin32_SystemResourcesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemResources, err error) {
	tmp, err := NewCIM_ComputerSystemResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemResources{
		CIM_ComputerSystemResource: tmp,
	}
	return
}
