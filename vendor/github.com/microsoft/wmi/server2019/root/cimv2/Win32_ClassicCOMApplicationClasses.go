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

// Win32_ClassicCOMApplicationClasses struct
type Win32_ClassicCOMApplicationClasses struct {
	*Win32_COMApplicationClasses
}

func NewWin32_ClassicCOMApplicationClassesEx1(instance *cim.WmiInstance) (newInstance *Win32_ClassicCOMApplicationClasses, err error) {
	tmp, err := NewWin32_COMApplicationClassesEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassicCOMApplicationClasses{
		Win32_COMApplicationClasses: tmp,
	}
	return
}

func NewWin32_ClassicCOMApplicationClassesEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ClassicCOMApplicationClasses, err error) {
	tmp, err := NewWin32_COMApplicationClassesEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ClassicCOMApplicationClasses{
		Win32_COMApplicationClasses: tmp,
	}
	return
}
