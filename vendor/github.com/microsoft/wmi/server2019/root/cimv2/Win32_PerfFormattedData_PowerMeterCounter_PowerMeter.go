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

// Win32_PerfFormattedData_PowerMeterCounter_PowerMeter struct
type Win32_PerfFormattedData_PowerMeterCounter_PowerMeter struct {
	*Win32_PerfFormattedData

	//
	Power uint32

	//
	PowerBudget uint32
}

func NewWin32_PerfFormattedData_PowerMeterCounter_PowerMeterEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PowerMeterCounter_PowerMeter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PowerMeterCounter_PowerMeter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PowerMeterCounter_PowerMeterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PowerMeterCounter_PowerMeter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PowerMeterCounter_PowerMeter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetPower sets the value of Power for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_PowerMeter) SetPropertyPower(value uint32) (err error) {
	return instance.SetProperty("Power", (value))
}

// GetPower gets the value of Power for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_PowerMeter) GetPropertyPower() (value uint32, err error) {
	retValue, err := instance.GetProperty("Power")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetPowerBudget sets the value of PowerBudget for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_PowerMeter) SetPropertyPowerBudget(value uint32) (err error) {
	return instance.SetProperty("PowerBudget", (value))
}

// GetPowerBudget gets the value of PowerBudget for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_PowerMeter) GetPropertyPowerBudget() (value uint32, err error) {
	retValue, err := instance.GetProperty("PowerBudget")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}
