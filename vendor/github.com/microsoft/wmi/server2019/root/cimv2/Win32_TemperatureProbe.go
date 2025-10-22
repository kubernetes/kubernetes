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

// Win32_TemperatureProbe struct
type Win32_TemperatureProbe struct {
	*CIM_TemperatureSensor
}

func NewWin32_TemperatureProbeEx1(instance *cim.WmiInstance) (newInstance *Win32_TemperatureProbe, err error) {
	tmp, err := NewCIM_TemperatureSensorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_TemperatureProbe{
		CIM_TemperatureSensor: tmp,
	}
	return
}

func NewWin32_TemperatureProbeEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_TemperatureProbe, err error) {
	tmp, err := NewCIM_TemperatureSensorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_TemperatureProbe{
		CIM_TemperatureSensor: tmp,
	}
	return
}
