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

// CIM_VoltageSensor struct
type CIM_VoltageSensor struct {
	*CIM_NumericSensor
}

func NewCIM_VoltageSensorEx1(instance *cim.WmiInstance) (newInstance *CIM_VoltageSensor, err error) {
	tmp, err := NewCIM_NumericSensorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_VoltageSensor{
		CIM_NumericSensor: tmp,
	}
	return
}

func NewCIM_VoltageSensorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_VoltageSensor, err error) {
	tmp, err := NewCIM_NumericSensorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_VoltageSensor{
		CIM_NumericSensor: tmp,
	}
	return
}
