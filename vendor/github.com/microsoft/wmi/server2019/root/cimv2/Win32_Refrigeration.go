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

// Win32_Refrigeration struct
type Win32_Refrigeration struct {
	*CIM_Refrigeration
}

func NewWin32_RefrigerationEx1(instance *cim.WmiInstance) (newInstance *Win32_Refrigeration, err error) {
	tmp, err := NewCIM_RefrigerationEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Refrigeration{
		CIM_Refrigeration: tmp,
	}
	return
}

func NewWin32_RefrigerationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Refrigeration, err error) {
	tmp, err := NewCIM_RefrigerationEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Refrigeration{
		CIM_Refrigeration: tmp,
	}
	return
}
