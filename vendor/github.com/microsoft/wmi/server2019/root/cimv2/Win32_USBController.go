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

// Win32_USBController struct
type Win32_USBController struct {
	*CIM_USBController
}

func NewWin32_USBControllerEx1(instance *cim.WmiInstance) (newInstance *Win32_USBController, err error) {
	tmp, err := NewCIM_USBControllerEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_USBController{
		CIM_USBController: tmp,
	}
	return
}

func NewWin32_USBControllerEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_USBController, err error) {
	tmp, err := NewCIM_USBControllerEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_USBController{
		CIM_USBController: tmp,
	}
	return
}
