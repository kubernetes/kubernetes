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

// CIM_DirectoryContainsFile struct
type CIM_DirectoryContainsFile struct {
	*CIM_Component
}

func NewCIM_DirectoryContainsFileEx1(instance *cim.WmiInstance) (newInstance *CIM_DirectoryContainsFile, err error) {
	tmp, err := NewCIM_ComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DirectoryContainsFile{
		CIM_Component: tmp,
	}
	return
}

func NewCIM_DirectoryContainsFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DirectoryContainsFile, err error) {
	tmp, err := NewCIM_ComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DirectoryContainsFile{
		CIM_Component: tmp,
	}
	return
}
