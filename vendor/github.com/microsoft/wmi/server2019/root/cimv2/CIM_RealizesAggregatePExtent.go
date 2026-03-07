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

// CIM_RealizesAggregatePExtent struct
type CIM_RealizesAggregatePExtent struct {
	*CIM_Realizes
}

func NewCIM_RealizesAggregatePExtentEx1(instance *cim.WmiInstance) (newInstance *CIM_RealizesAggregatePExtent, err error) {
	tmp, err := NewCIM_RealizesEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_RealizesAggregatePExtent{
		CIM_Realizes: tmp,
	}
	return
}

func NewCIM_RealizesAggregatePExtentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_RealizesAggregatePExtent, err error) {
	tmp, err := NewCIM_RealizesEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_RealizesAggregatePExtent{
		CIM_Realizes: tmp,
	}
	return
}
