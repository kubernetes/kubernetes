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

// CIM_OperatingSystemSoftwareFeature struct
type CIM_OperatingSystemSoftwareFeature struct {
	*CIM_Component
}

func NewCIM_OperatingSystemSoftwareFeatureEx1(instance *cim.WmiInstance) (newInstance *CIM_OperatingSystemSoftwareFeature, err error) {
	tmp, err := NewCIM_ComponentEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_OperatingSystemSoftwareFeature{
		CIM_Component: tmp,
	}
	return
}

func NewCIM_OperatingSystemSoftwareFeatureEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_OperatingSystemSoftwareFeature, err error) {
	tmp, err := NewCIM_ComponentEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_OperatingSystemSoftwareFeature{
		CIM_Component: tmp,
	}
	return
}
