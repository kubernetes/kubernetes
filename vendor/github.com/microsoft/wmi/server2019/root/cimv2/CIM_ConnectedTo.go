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

// CIM_ConnectedTo struct
type CIM_ConnectedTo struct {
	*CIM_Dependency
}

func NewCIM_ConnectedToEx1(instance *cim.WmiInstance) (newInstance *CIM_ConnectedTo, err error) {
	tmp, err := NewCIM_DependencyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ConnectedTo{
		CIM_Dependency: tmp,
	}
	return
}

func NewCIM_ConnectedToEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ConnectedTo, err error) {
	tmp, err := NewCIM_DependencyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ConnectedTo{
		CIM_Dependency: tmp,
	}
	return
}
