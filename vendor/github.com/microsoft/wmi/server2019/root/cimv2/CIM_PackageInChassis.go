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

// CIM_PackageInChassis struct
type CIM_PackageInChassis struct {
	*CIM_Container
}

func NewCIM_PackageInChassisEx1(instance *cim.WmiInstance) (newInstance *CIM_PackageInChassis, err error) {
	tmp, err := NewCIM_ContainerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_PackageInChassis{
		CIM_Container: tmp,
	}
	return
}

func NewCIM_PackageInChassisEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_PackageInChassis, err error) {
	tmp, err := NewCIM_ContainerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_PackageInChassis{
		CIM_Container: tmp,
	}
	return
}
