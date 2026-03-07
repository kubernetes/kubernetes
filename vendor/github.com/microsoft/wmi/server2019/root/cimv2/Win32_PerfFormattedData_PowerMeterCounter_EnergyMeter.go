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

// Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter struct
type Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter struct {
	*Win32_PerfFormattedData

	//
	Energy uint64

	//
	Power uint64

	//
	Time uint64
}

func NewWin32_PerfFormattedData_PowerMeterCounter_EnergyMeterEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_PowerMeterCounter_EnergyMeterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetEnergy sets the value of Energy for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter) SetPropertyEnergy(value uint64) (err error) {
	return instance.SetProperty("Energy", (value))
}

// GetEnergy gets the value of Energy for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter) GetPropertyEnergy() (value uint64, err error) {
	retValue, err := instance.GetProperty("Energy")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPower sets the value of Power for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter) SetPropertyPower(value uint64) (err error) {
	return instance.SetProperty("Power", (value))
}

// GetPower gets the value of Power for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter) GetPropertyPower() (value uint64, err error) {
	retValue, err := instance.GetProperty("Power")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetTime sets the value of Time for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter) SetPropertyTime(value uint64) (err error) {
	return instance.SetProperty("Time", (value))
}

// GetTime gets the value of Time for the instance
func (instance *Win32_PerfFormattedData_PowerMeterCounter_EnergyMeter) GetPropertyTime() (value uint64, err error) {
	retValue, err := instance.GetProperty("Time")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}
