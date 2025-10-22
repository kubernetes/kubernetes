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

// CIM_PSExtentBasedOnPExtent struct
type CIM_PSExtentBasedOnPExtent struct {
	*CIM_BasedOn
}

func NewCIM_PSExtentBasedOnPExtentEx1(instance *cim.WmiInstance) (newInstance *CIM_PSExtentBasedOnPExtent, err error) {
	tmp, err := NewCIM_BasedOnEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PSExtentBasedOnPExtent{
		CIM_BasedOn: tmp,
	}
	return
}

func NewCIM_PSExtentBasedOnPExtentEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PSExtentBasedOnPExtent, err error) {
	tmp, err := NewCIM_BasedOnEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PSExtentBasedOnPExtent{
		CIM_BasedOn: tmp,
	}
	return
}
