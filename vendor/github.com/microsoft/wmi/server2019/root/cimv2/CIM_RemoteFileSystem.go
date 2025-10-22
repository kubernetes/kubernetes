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

// CIM_RemoteFileSystem struct
type CIM_RemoteFileSystem struct {
	*CIM_FileSystem
}

func NewCIM_RemoteFileSystemEx1(instance *cim.WmiInstance) (newInstance *CIM_RemoteFileSystem, err error) {
	tmp, err := NewCIM_FileSystemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_RemoteFileSystem{
		CIM_FileSystem: tmp,
	}
	return
}

func NewCIM_RemoteFileSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_RemoteFileSystem, err error) {
	tmp, err := NewCIM_FileSystemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_RemoteFileSystem{
		CIM_FileSystem: tmp,
	}
	return
}
