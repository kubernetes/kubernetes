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

// CIM_BootOSFromFS struct
type CIM_BootOSFromFS struct {
	*CIM_Dependency
}

func NewCIM_BootOSFromFSEx1(instance *cim.WmiInstance) (newInstance *CIM_BootOSFromFS, err error) {
	tmp, err := NewCIM_DependencyEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_BootOSFromFS{
		CIM_Dependency: tmp,
	}
	return
}

func NewCIM_BootOSFromFSEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_BootOSFromFS, err error) {
	tmp, err := NewCIM_DependencyEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_BootOSFromFS{
		CIM_Dependency: tmp,
	}
	return
}
