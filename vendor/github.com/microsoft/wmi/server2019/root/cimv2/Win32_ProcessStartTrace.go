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

// Win32_ProcessStartTrace struct
type Win32_ProcessStartTrace struct {
	*Win32_ProcessTrace
}

func NewWin32_ProcessStartTraceEx1(instance *cim.WmiInstance) (newInstance *Win32_ProcessStartTrace, err error) {
	tmp, err := NewWin32_ProcessTraceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ProcessStartTrace{
		Win32_ProcessTrace: tmp,
	}
	return
}

func NewWin32_ProcessStartTraceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ProcessStartTrace, err error) {
	tmp, err := NewWin32_ProcessTraceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ProcessStartTrace{
		Win32_ProcessTrace: tmp,
	}
	return
}
