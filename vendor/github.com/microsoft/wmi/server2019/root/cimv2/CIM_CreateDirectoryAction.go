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

// CIM_CreateDirectoryAction struct
type CIM_CreateDirectoryAction struct {
	*CIM_DirectoryAction
}

func NewCIM_CreateDirectoryActionEx1(instance *cim.WmiInstance) (newInstance *CIM_CreateDirectoryAction, err error) {
	tmp, err := NewCIM_DirectoryActionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_CreateDirectoryAction{
		CIM_DirectoryAction: tmp,
	}
	return
}

func NewCIM_CreateDirectoryActionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_CreateDirectoryAction, err error) {
	tmp, err := NewCIM_DirectoryActionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_CreateDirectoryAction{
		CIM_DirectoryAction: tmp,
	}
	return
}
