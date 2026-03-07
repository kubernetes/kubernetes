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

// CIM_SpareGroup struct
type CIM_SpareGroup struct {
	*CIM_RedundancyGroup
}

func NewCIM_SpareGroupEx1(instance *cim.WmiInstance) (newInstance *CIM_SpareGroup, err error) {
	tmp, err := NewCIM_RedundancyGroupEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_SpareGroup{
		CIM_RedundancyGroup: tmp,
	}
	return
}

func NewCIM_SpareGroupEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SpareGroup, err error) {
	tmp, err := NewCIM_RedundancyGroupEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SpareGroup{
		CIM_RedundancyGroup: tmp,
	}
	return
}
