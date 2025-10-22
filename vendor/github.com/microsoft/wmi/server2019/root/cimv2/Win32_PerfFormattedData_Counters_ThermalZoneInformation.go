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

// Win32_PerfFormattedData_Counters_ThermalZoneInformation struct
type Win32_PerfFormattedData_Counters_ThermalZoneInformation struct {
	*Win32_PerfFormattedData

	//
	HighPrecisionTemperature uint32

	//
	PercentPassiveLimit uint32

	//
	Temperature uint32

	//
	ThrottleReasons uint32
}

func NewWin32_PerfFormattedData_Counters_ThermalZoneInformationEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Counters_ThermalZoneInformation, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_ThermalZoneInformation{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Counters_ThermalZoneInformationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Counters_ThermalZoneInformation, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Counters_ThermalZoneInformation{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetHighPrecisionTemperature sets the value of HighPrecisionTemperature for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) SetPropertyHighPrecisionTemperature(value uint32) (err error) {
	return instance.SetProperty("HighPrecisionTemperature", (value))
}

// GetHighPrecisionTemperature gets the value of HighPrecisionTemperature for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) GetPropertyHighPrecisionTemperature() (value uint32, err error) {
	retValue, err := instance.GetProperty("HighPrecisionTemperature")
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

// SetPercentPassiveLimit sets the value of PercentPassiveLimit for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) SetPropertyPercentPassiveLimit(value uint32) (err error) {
	return instance.SetProperty("PercentPassiveLimit", (value))
}

// GetPercentPassiveLimit gets the value of PercentPassiveLimit for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) GetPropertyPercentPassiveLimit() (value uint32, err error) {
	retValue, err := instance.GetProperty("PercentPassiveLimit")
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

// SetTemperature sets the value of Temperature for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) SetPropertyTemperature(value uint32) (err error) {
	return instance.SetProperty("Temperature", (value))
}

// GetTemperature gets the value of Temperature for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) GetPropertyTemperature() (value uint32, err error) {
	retValue, err := instance.GetProperty("Temperature")
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

// SetThrottleReasons sets the value of ThrottleReasons for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) SetPropertyThrottleReasons(value uint32) (err error) {
	return instance.SetProperty("ThrottleReasons", (value))
}

// GetThrottleReasons gets the value of ThrottleReasons for the instance
func (instance *Win32_PerfFormattedData_Counters_ThermalZoneInformation) GetPropertyThrottleReasons() (value uint32, err error) {
	retValue, err := instance.GetProperty("ThrottleReasons")
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
