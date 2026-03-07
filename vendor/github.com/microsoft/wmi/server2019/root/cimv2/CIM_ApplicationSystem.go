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

// CIM_ApplicationSystem struct
type CIM_ApplicationSystem struct {
	*CIM_System
}

func NewCIM_ApplicationSystemEx1(instance *cim.WmiInstance) (newInstance *CIM_ApplicationSystem, err error) {
	tmp, err := NewCIM_SystemEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ApplicationSystem{
		CIM_System: tmp,
	}
	return
}

func NewCIM_ApplicationSystemEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ApplicationSystem, err error) {
	tmp, err := NewCIM_SystemEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ApplicationSystem{
		CIM_System: tmp,
	}
	return
}
