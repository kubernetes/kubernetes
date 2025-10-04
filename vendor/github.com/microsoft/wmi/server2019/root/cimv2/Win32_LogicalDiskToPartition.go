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

// Win32_LogicalDiskToPartition struct
type Win32_LogicalDiskToPartition struct {
	*CIM_LogicalDiskBasedOnPartition
}

func NewWin32_LogicalDiskToPartitionEx1(instance *cim.WmiInstance) (newInstance *Win32_LogicalDiskToPartition, err error) {
	tmp, err := NewCIM_LogicalDiskBasedOnPartitionEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalDiskToPartition{
		CIM_LogicalDiskBasedOnPartition: tmp,
	}
	return
}

func NewWin32_LogicalDiskToPartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_LogicalDiskToPartition, err error) {
	tmp, err := NewCIM_LogicalDiskBasedOnPartitionEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_LogicalDiskToPartition{
		CIM_LogicalDiskBasedOnPartition: tmp,
	}
	return
}
