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

// Win32_ComputerSystemProcessor struct
type Win32_ComputerSystemProcessor struct {
	*Win32_SystemDevices
}

func NewWin32_ComputerSystemProcessorEx1(instance *cim.WmiInstance) (newInstance *Win32_ComputerSystemProcessor, err error) {
	tmp, err := NewWin32_SystemDevicesEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystemProcessor{
		Win32_SystemDevices: tmp,
	}
	return
}

func NewWin32_ComputerSystemProcessorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_ComputerSystemProcessor, err error) {
	tmp, err := NewWin32_SystemDevicesEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_ComputerSystemProcessor{
		Win32_SystemDevices: tmp,
	}
	return
}
