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

// Win32_SystemTrace struct
type Win32_SystemTrace struct {
	*__ExtrinsicEvent
}

func NewWin32_SystemTraceEx1(instance *cim.WmiInstance) (newInstance *Win32_SystemTrace, err error) {
	tmp, err := New__ExtrinsicEventEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemTrace{
		__ExtrinsicEvent: tmp,
	}
	return
}

func NewWin32_SystemTraceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_SystemTrace, err error) {
	tmp, err := New__ExtrinsicEventEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_SystemTrace{
		__ExtrinsicEvent: tmp,
	}
	return
}
