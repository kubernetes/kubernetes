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

// Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity struct
type Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity struct {
	*Win32_PerfRawData

	//
	DevicePowerState uint32

	//
	LowPowerTransitionsLifetime uint32

	//
	PercentTimeSuspendedInstantaneous uint64

	//
	PercentTimeSuspendedLifetime uint64

	//
	PercentTimeSuspendedLifetime_Base uint64
}

func NewWin32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivityEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivityEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDevicePowerState sets the value of DevicePowerState for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) SetPropertyDevicePowerState(value uint32) (err error) {
	return instance.SetProperty("DevicePowerState", (value))
}

// GetDevicePowerState gets the value of DevicePowerState for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) GetPropertyDevicePowerState() (value uint32, err error) {
	retValue, err := instance.GetProperty("DevicePowerState")
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

// SetLowPowerTransitionsLifetime sets the value of LowPowerTransitionsLifetime for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) SetPropertyLowPowerTransitionsLifetime(value uint32) (err error) {
	return instance.SetProperty("LowPowerTransitionsLifetime", (value))
}

// GetLowPowerTransitionsLifetime gets the value of LowPowerTransitionsLifetime for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) GetPropertyLowPowerTransitionsLifetime() (value uint32, err error) {
	retValue, err := instance.GetProperty("LowPowerTransitionsLifetime")
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

// SetPercentTimeSuspendedInstantaneous sets the value of PercentTimeSuspendedInstantaneous for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) SetPropertyPercentTimeSuspendedInstantaneous(value uint64) (err error) {
	return instance.SetProperty("PercentTimeSuspendedInstantaneous", (value))
}

// GetPercentTimeSuspendedInstantaneous gets the value of PercentTimeSuspendedInstantaneous for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) GetPropertyPercentTimeSuspendedInstantaneous() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentTimeSuspendedInstantaneous")
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

// SetPercentTimeSuspendedLifetime sets the value of PercentTimeSuspendedLifetime for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) SetPropertyPercentTimeSuspendedLifetime(value uint64) (err error) {
	return instance.SetProperty("PercentTimeSuspendedLifetime", (value))
}

// GetPercentTimeSuspendedLifetime gets the value of PercentTimeSuspendedLifetime for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) GetPropertyPercentTimeSuspendedLifetime() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentTimeSuspendedLifetime")
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

// SetPercentTimeSuspendedLifetime_Base sets the value of PercentTimeSuspendedLifetime_Base for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) SetPropertyPercentTimeSuspendedLifetime_Base(value uint64) (err error) {
	return instance.SetProperty("PercentTimeSuspendedLifetime_Base", (value))
}

// GetPercentTimeSuspendedLifetime_Base gets the value of PercentTimeSuspendedLifetime_Base for the instance
func (instance *Win32_PerfRawData_Counters_PhysicalNetworkInterfaceCardActivity) GetPropertyPercentTimeSuspendedLifetime_Base() (value uint64, err error) {
	retValue, err := instance.GetProperty("PercentTimeSuspendedLifetime_Base")
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
