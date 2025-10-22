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

// CIM_ComputerSystemMappedIO struct
type CIM_ComputerSystemMappedIO struct {
	*CIM_ComputerSystemResource
}

func NewCIM_ComputerSystemMappedIOEx1(instance *cim.WmiInstance) (newInstance *CIM_ComputerSystemMappedIO, err error) {
	tmp, err := NewCIM_ComputerSystemResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_ComputerSystemMappedIO{
		CIM_ComputerSystemResource: tmp,
	}
	return
}

func NewCIM_ComputerSystemMappedIOEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_ComputerSystemMappedIO, err error) {
	tmp, err := NewCIM_ComputerSystemResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_ComputerSystemMappedIO{
		CIM_ComputerSystemResource: tmp,
	}
	return
}
