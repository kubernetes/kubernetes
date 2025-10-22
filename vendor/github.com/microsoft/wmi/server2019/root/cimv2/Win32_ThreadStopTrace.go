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

// Win32_ThreadStopTrace struct
type Win32_ThreadStopTrace struct {
	*Win32_ThreadTrace
}

func NewWin32_ThreadStopTraceEx1(instance *cim.WmiInstance) (newInstance *Win32_ThreadStopTrace, err error) {
	tmp, err := NewWin32_ThreadTraceEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ThreadStopTrace{
		Win32_ThreadTrace: tmp,
	}
	return
}

func NewWin32_ThreadStopTraceEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ThreadStopTrace, err error) {
	tmp, err := NewWin32_ThreadTraceEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ThreadStopTrace{
		Win32_ThreadTrace: tmp,
	}
	return
}
