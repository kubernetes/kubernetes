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

// CIM_PExtentRedundancyComponent struct
type CIM_PExtentRedundancyComponent struct {
	*CIM_RedundancyComponent
}

func NewCIM_PExtentRedundancyComponentEx1(instance *cim.WmiInstance) (newInstance *CIM_PExtentRedundancyComponent, err error) {
	tmp, err := NewCIM_RedundancyComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PExtentRedundancyComponent{
		CIM_RedundancyComponent: tmp,
	}
	return
}

func NewCIM_PExtentRedundancyComponentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PExtentRedundancyComponent, err error) {
	tmp, err := NewCIM_RedundancyComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PExtentRedundancyComponent{
		CIM_RedundancyComponent: tmp,
	}
	return
}
