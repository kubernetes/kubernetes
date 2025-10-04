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

// CIM_LogicalDiskBasedOnPartition struct
type CIM_LogicalDiskBasedOnPartition struct {
	*CIM_BasedOn
}

func NewCIM_LogicalDiskBasedOnPartitionEx1(instance *cim.WmiInstance) (newInstance *CIM_LogicalDiskBasedOnPartition, err error) {
	tmp, err := NewCIM_BasedOnEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalDiskBasedOnPartition{
		CIM_BasedOn: tmp,
	}
	return
}

func NewCIM_LogicalDiskBasedOnPartitionEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_LogicalDiskBasedOnPartition, err error) {
	tmp, err := NewCIM_BasedOnEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalDiskBasedOnPartition{
		CIM_BasedOn: tmp,
	}
	return
}
