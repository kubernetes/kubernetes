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

// CIM_DeviceFile struct
type CIM_DeviceFile struct {
	*CIM_LogicalFile
}

func NewCIM_DeviceFileEx1(instance *cim.WmiInstance) (newInstance *CIM_DeviceFile, err error) {
	tmp, err := NewCIM_LogicalFileEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_DeviceFile{
		CIM_LogicalFile: tmp,
	}
	return
}

func NewCIM_DeviceFileEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_DeviceFile, err error) {
	tmp, err := NewCIM_LogicalFileEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_DeviceFile{
		CIM_LogicalFile: tmp,
	}
	return
}
