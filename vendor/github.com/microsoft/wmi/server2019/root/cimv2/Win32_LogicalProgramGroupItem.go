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

// Win32_LogicalProgramGroupItem struct
type Win32_LogicalProgramGroupItem struct {
	*Win32_ProgramGroupOrItem
}

func NewWin32_LogicalProgramGroupItemEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalProgramGroupItem, err error) {
	tmp, err := NewWin32_ProgramGroupOrItemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalProgramGroupItem{
		Win32_ProgramGroupOrItem: tmp,
	}
	return
}

func NewWin32_LogicalProgramGroupItemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalProgramGroupItem, err error) {
	tmp, err := NewWin32_ProgramGroupOrItemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalProgramGroupItem{
		Win32_ProgramGroupOrItem: tmp,
	}
	return
}
