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

// Win32_SessionProcess struct
type Win32_SessionProcess struct {
	*Win32_SessionResource
}

func NewWin32_SessionProcessEx1(instance *cim.WmiInstance) (newInstance *Win32_SessionProcess, err error) {
	tmp, err := NewWin32_SessionResourceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SessionProcess{
		Win32_SessionResource: tmp,
	}
	return
}

func NewWin32_SessionProcessEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SessionProcess, err error) {
	tmp, err := NewWin32_SessionResourceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SessionProcess{
		Win32_SessionResource: tmp,
	}
	return
}
