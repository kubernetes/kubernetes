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

// CIM_LogicalElement struct
type CIM_LogicalElement struct {
	*CIM_ManagedSystemElement
}

func NewCIM_LogicalElementEx1(instance *cim.WmiInstance) (newInstance *CIM_LogicalElement, err error) {
	tmp, err := NewCIM_ManagedSystemElementEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalElement{
		CIM_ManagedSystemElement: tmp,
	}
	return
}

func NewCIM_LogicalElementEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_LogicalElement, err error) {
	tmp, err := NewCIM_ManagedSystemElementEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_LogicalElement{
		CIM_ManagedSystemElement: tmp,
	}
	return
}
