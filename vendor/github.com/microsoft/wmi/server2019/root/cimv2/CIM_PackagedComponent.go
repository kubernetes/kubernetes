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

// CIM_PackagedComponent struct
type CIM_PackagedComponent struct {
	*CIM_Container
}

func NewCIM_PackagedComponentEx1(instance *cim.WmiInstance) (newInstance *CIM_PackagedComponent, err error) {
	tmp, err := NewCIM_ContainerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PackagedComponent{
		CIM_Container: tmp,
	}
	return
}

func NewCIM_PackagedComponentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PackagedComponent, err error) {
	tmp, err := NewCIM_ContainerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PackagedComponent{
		CIM_Container: tmp,
	}
	return
}
