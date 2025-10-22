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

// CIM_HostedBootService struct
type CIM_HostedBootService struct {
	*CIM_HostedService
}

func NewCIM_HostedBootServiceEx1(instance *cim.WmiInstance) (newInstance *CIM_HostedBootService, err error) {
	tmp, err := NewCIM_HostedServiceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_HostedBootService{
		CIM_HostedService: tmp,
	}
	return
}

func NewCIM_HostedBootServiceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_HostedBootService, err error) {
	tmp, err := NewCIM_HostedServiceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_HostedBootService{
		CIM_HostedService: tmp,
	}
	return
}
