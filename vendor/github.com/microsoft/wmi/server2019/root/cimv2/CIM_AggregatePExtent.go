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

// CIM_AggregatePExtent struct
type CIM_AggregatePExtent struct {
	*CIM_StorageExtent
}

func NewCIM_AggregatePExtentEx1(instance *cim.WmiInstance) (newInstance *CIM_AggregatePExtent, err error) {
	tmp, err := NewCIM_StorageExtentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_AggregatePExtent{
		CIM_StorageExtent: tmp,
	}
	return
}

func NewCIM_AggregatePExtentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_AggregatePExtent, err error) {
	tmp, err := NewCIM_StorageExtentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_AggregatePExtent{
		CIM_StorageExtent: tmp,
	}
	return
}
