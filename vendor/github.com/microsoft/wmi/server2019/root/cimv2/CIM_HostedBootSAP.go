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

// CIM_HostedBootSAP struct
type CIM_HostedBootSAP struct {
	*CIM_HostedAccessPoint
}

func NewCIM_HostedBootSAPEx1(instance *cim.WmiInstance) (newInstance *CIM_HostedBootSAP, err error) {
	tmp, err := NewCIM_HostedAccessPointEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_HostedBootSAP{
		CIM_HostedAccessPoint: tmp,
	}
	return
}

func NewCIM_HostedBootSAPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_HostedBootSAP, err error) {
	tmp, err := NewCIM_HostedAccessPointEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_HostedBootSAP{
		CIM_HostedAccessPoint: tmp,
	}
	return
}
