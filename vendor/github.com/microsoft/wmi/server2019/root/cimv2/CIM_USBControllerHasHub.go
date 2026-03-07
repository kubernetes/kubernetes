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

// CIM_USBControllerHasHub struct
type CIM_USBControllerHasHub struct {
	*CIM_ControlledBy
}

func NewCIM_USBControllerHasHubEx1(instance *cim.WmiInstance) (newInstance *CIM_USBControllerHasHub, err error) {
	tmp, err := NewCIM_ControlledByEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_USBControllerHasHub{
		CIM_ControlledBy: tmp,
	}
	return
}

func NewCIM_USBControllerHasHubEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_USBControllerHasHub, err error) {
	tmp, err := NewCIM_ControlledByEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_USBControllerHasHub{
		CIM_ControlledBy: tmp,
	}
	return
}
