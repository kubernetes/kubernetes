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

// Win32_PNPAllocatedResource struct
type Win32_PNPAllocatedResource struct {
	*CIM_AllocatedResource
}

func NewWin32_PNPAllocatedResourceEx1(instance *cim.WmiInstance) (newInstance *Win32_PNPAllocatedResource, err error) {
	tmp, err := NewCIM_AllocatedResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PNPAllocatedResource{
		CIM_AllocatedResource: tmp,
	}
	return
}

func NewWin32_PNPAllocatedResourceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PNPAllocatedResource, err error) {
	tmp, err := NewCIM_AllocatedResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PNPAllocatedResource{
		CIM_AllocatedResource: tmp,
	}
	return
}
