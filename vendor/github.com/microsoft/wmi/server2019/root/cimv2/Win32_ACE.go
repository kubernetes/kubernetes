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

// Win32_ACE struct
type Win32_ACE struct {
	*__ACE
}

func NewWin32_ACEEx1(instance *cim.WmiInstance) (newInstance *Win32_ACE, err error) {
	tmp, err := New__ACEEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ACE{
		__ACE: tmp,
	}
	return
}

func NewWin32_ACEEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ACE, err error) {
	tmp, err := New__ACEEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ACE{
		__ACE: tmp,
	}
	return
}
