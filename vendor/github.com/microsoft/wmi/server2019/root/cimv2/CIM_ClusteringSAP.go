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

// CIM_ClusteringSAP struct
type CIM_ClusteringSAP struct {
	*CIM_ServiceAccessPoint
}

func NewCIM_ClusteringSAPEx1(instance *cim.WmiInstance) (newInstance *CIM_ClusteringSAP, err error) {
	tmp, err := NewCIM_ServiceAccessPointEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ClusteringSAP{
		CIM_ServiceAccessPoint: tmp,
	}
	return
}

func NewCIM_ClusteringSAPEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ClusteringSAP, err error) {
	tmp, err := NewCIM_ServiceAccessPointEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ClusteringSAP{
		CIM_ServiceAccessPoint: tmp,
	}
	return
}
