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

// CIM_HostedFileSystem struct
type CIM_HostedFileSystem struct {
	*CIM_SystemComponent
}

func NewCIM_HostedFileSystemEx1(instance *cim.WmiInstance) (newInstance *CIM_HostedFileSystem, err error) {
	tmp, err := NewCIM_SystemComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_HostedFileSystem{
		CIM_SystemComponent: tmp,
	}
	return
}

func NewCIM_HostedFileSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_HostedFileSystem, err error) {
	tmp, err := NewCIM_SystemComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_HostedFileSystem{
		CIM_SystemComponent: tmp,
	}
	return
}
