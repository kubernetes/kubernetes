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

// CIM_RedundancyComponent struct
type CIM_RedundancyComponent struct {
	*CIM_Component
}

func NewCIM_RedundancyComponentEx1(instance *cim.WmiInstance) (newInstance *CIM_RedundancyComponent, err error) {
	tmp, err := NewCIM_ComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_RedundancyComponent{
		CIM_Component: tmp,
	}
	return
}

func NewCIM_RedundancyComponentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_RedundancyComponent, err error) {
	tmp, err := NewCIM_ComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_RedundancyComponent{
		CIM_Component: tmp,
	}
	return
}
