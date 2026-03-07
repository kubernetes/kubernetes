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

// Win32_SecurityDescriptor struct
type Win32_SecurityDescriptor struct {
	*__SecurityDescriptor
}

func NewWin32_SecurityDescriptorEx1(instance *cim.WmiInstance) (newInstance *Win32_SecurityDescriptor, err error) {
	tmp, err := New__SecurityDescriptorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SecurityDescriptor{
		__SecurityDescriptor: tmp,
	}
	return
}

func NewWin32_SecurityDescriptorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SecurityDescriptor, err error) {
	tmp, err := New__SecurityDescriptorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SecurityDescriptor{
		__SecurityDescriptor: tmp,
	}
	return
}
