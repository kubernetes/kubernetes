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

// CIM_MultiStateSensor struct
type CIM_MultiStateSensor struct {
	*CIM_Sensor
}

func NewCIM_MultiStateSensorEx1(instance *cim.WmiInstance) (newInstance *CIM_MultiStateSensor, err error) {
	tmp, err := NewCIM_SensorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_MultiStateSensor{
		CIM_Sensor: tmp,
	}
	return
}

func NewCIM_MultiStateSensorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_MultiStateSensor, err error) {
	tmp, err := NewCIM_SensorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_MultiStateSensor{
		CIM_Sensor: tmp,
	}
	return
}
