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

// CIM_ErrorCountersForDevice struct
type CIM_ErrorCountersForDevice struct {
	*CIM_Statistics
}

func NewCIM_ErrorCountersForDeviceEx1(instance *cim.WmiInstance) (newInstance *CIM_ErrorCountersForDevice, err error) {
	tmp, err := NewCIM_StatisticsEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ErrorCountersForDevice{
		CIM_Statistics: tmp,
	}
	return
}

func NewCIM_ErrorCountersForDeviceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ErrorCountersForDevice, err error) {
	tmp, err := NewCIM_StatisticsEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ErrorCountersForDevice{
		CIM_Statistics: tmp,
	}
	return
}
