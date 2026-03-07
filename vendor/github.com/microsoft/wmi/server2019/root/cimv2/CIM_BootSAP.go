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

// CIM_BootSAP struct
type CIM_BootSAP struct {
	*CIM_ServiceAccessPoint
}

func NewCIM_BootSAPEx1(instance *cim.WmiInstance) (newInstance *CIM_BootSAP, err error) {
	tmp, err := NewCIM_ServiceAccessPointEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_BootSAP{
		CIM_ServiceAccessPoint: tmp,
	}
	return
}

func NewCIM_BootSAPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_BootSAP, err error) {
	tmp, err := NewCIM_ServiceAccessPointEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_BootSAP{
		CIM_ServiceAccessPoint: tmp,
	}
	return
}
