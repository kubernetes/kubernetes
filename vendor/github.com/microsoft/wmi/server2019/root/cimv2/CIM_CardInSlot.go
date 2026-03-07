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

// CIM_CardInSlot struct
type CIM_CardInSlot struct {
	*CIM_PackageInSlot
}

func NewCIM_CardInSlotEx1(instance *cim.WmiInstance) (newInstance *CIM_CardInSlot, err error) {
	tmp, err := NewCIM_PackageInSlotEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_CardInSlot{
		CIM_PackageInSlot: tmp,
	}
	return
}

func NewCIM_CardInSlotEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_CardInSlot, err error) {
	tmp, err := NewCIM_PackageInSlotEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_CardInSlot{
		CIM_PackageInSlot: tmp,
	}
	return
}
