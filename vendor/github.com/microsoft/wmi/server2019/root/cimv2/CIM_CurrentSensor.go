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

// CIM_CurrentSensor struct
type CIM_CurrentSensor struct {
	*CIM_NumericSensor
}

func NewCIM_CurrentSensorEx1(instance *cim.WmiInstance) (newInstance *CIM_CurrentSensor, err error) {
	tmp, err := NewCIM_NumericSensorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_CurrentSensor{
		CIM_NumericSensor: tmp,
	}
	return
}

func NewCIM_CurrentSensorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_CurrentSensor, err error) {
	tmp, err := NewCIM_NumericSensorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_CurrentSensor{
		CIM_NumericSensor: tmp,
	}
	return
}
