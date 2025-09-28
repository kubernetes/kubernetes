// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/instance"
	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Win32_Reliability struct
type Win32_Reliability struct {
	*cim.WmiInstance
}

func NewWin32_ReliabilityEx1(instance *cim.WmiInstance) (newInstance *Win32_Reliability, err error) {
	tmp, err := instance, nil

	if err != nil {
		return
	}
	newInstance = &Win32_Reliability{
		WmiInstance: tmp,
	}
	return
}

func NewWin32_ReliabilityEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Reliability, err error) {
	tmp, err := instance.GetWmiInstance(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Reliability{
		WmiInstance: tmp,
	}
	return
}
