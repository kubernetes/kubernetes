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

// Win32_ODBCDriverSoftwareElement struct
type Win32_ODBCDriverSoftwareElement struct {
	*CIM_SoftwareElementChecks
}

func NewWin32_ODBCDriverSoftwareElementEx1(instance *cim.WmiInstance) (newInstance *Win32_ODBCDriverSoftwareElement, err error) {
	tmp, err := NewCIM_SoftwareElementChecksEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCDriverSoftwareElement{
		CIM_SoftwareElementChecks: tmp,
	}
	return
}

func NewWin32_ODBCDriverSoftwareElementEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ODBCDriverSoftwareElement, err error) {
	tmp, err := NewCIM_SoftwareElementChecksEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ODBCDriverSoftwareElement{
		CIM_SoftwareElementChecks: tmp,
	}
	return
}
