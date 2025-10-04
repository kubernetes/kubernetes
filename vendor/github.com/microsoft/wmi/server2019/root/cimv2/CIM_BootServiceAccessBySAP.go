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

// CIM_BootServiceAccessBySAP struct
type CIM_BootServiceAccessBySAP struct {
	*CIM_ServiceAccessBySAP
}

func NewCIM_BootServiceAccessBySAPEx1(instance *cim.WmiInstance) (newInstance *CIM_BootServiceAccessBySAP, err error) {
	tmp, err := NewCIM_ServiceAccessBySAPEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_BootServiceAccessBySAP{
		CIM_ServiceAccessBySAP: tmp,
	}
	return
}

func NewCIM_BootServiceAccessBySAPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_BootServiceAccessBySAP, err error) {
	tmp, err := NewCIM_ServiceAccessBySAPEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_BootServiceAccessBySAP{
		CIM_ServiceAccessBySAP: tmp,
	}
	return
}
