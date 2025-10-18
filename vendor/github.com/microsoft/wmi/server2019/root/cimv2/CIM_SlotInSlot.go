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

// CIM_SlotInSlot struct
type CIM_SlotInSlot struct {
	*CIM_ConnectedTo
}

func NewCIM_SlotInSlotEx1(instance *cim.WmiInstance) (newInstance *CIM_SlotInSlot, err error) {
	tmp, err := NewCIM_ConnectedToEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_SlotInSlot{
		CIM_ConnectedTo: tmp,
	}
	return
}

func NewCIM_SlotInSlotEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_SlotInSlot, err error) {
	tmp, err := NewCIM_ConnectedToEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_SlotInSlot{
		CIM_ConnectedTo: tmp,
	}
	return
}
