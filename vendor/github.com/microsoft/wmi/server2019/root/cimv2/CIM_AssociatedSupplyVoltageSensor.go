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
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// CIM_AssociatedSupplyVoltageSensor struct
type CIM_AssociatedSupplyVoltageSensor struct {
	*CIM_AssociatedSensor

	//
	MonitoringRange uint16
}

func NewCIM_AssociatedSupplyVoltageSensorEx1(instance *cim.WmiInstance) (newInstance *CIM_AssociatedSupplyVoltageSensor, err error) {
	tmp, err := NewCIM_AssociatedSensorEx1(instance)

	if err != nil {
		return
	}
	newInstance = &CIM_AssociatedSupplyVoltageSensor{
		CIM_AssociatedSensor: tmp,
	}
	return
}

func NewCIM_AssociatedSupplyVoltageSensorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *CIM_AssociatedSupplyVoltageSensor, err error) {
	tmp, err := NewCIM_AssociatedSensorEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &CIM_AssociatedSupplyVoltageSensor{
		CIM_AssociatedSensor: tmp,
	}
	return
}

// SetMonitoringRange sets the value of MonitoringRange for the instance
func (instance *CIM_AssociatedSupplyVoltageSensor) SetPropertyMonitoringRange(value uint16) (err error) {
	return instance.SetProperty("MonitoringRange", (value))
}

// GetMonitoringRange gets the value of MonitoringRange for the instance
func (instance *CIM_AssociatedSupplyVoltageSensor) GetPropertyMonitoringRange() (value uint16, err error) {
	retValue, err := instance.GetProperty("MonitoringRange")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}
