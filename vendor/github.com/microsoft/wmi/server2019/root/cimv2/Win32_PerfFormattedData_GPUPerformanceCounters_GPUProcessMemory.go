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

// Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory struct
type Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory struct {
	*Win32_PerfFormattedData

	//
	DedicatedUsage uint64

	//
	LocalUsage uint64

	//
	NonLocalUsage uint64

	//
	SharedUsage uint64

	//
	TotalCommitted uint64
}

func NewWin32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemoryEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetDedicatedUsage sets the value of DedicatedUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) SetPropertyDedicatedUsage(value uint64) (err error) {
	return instance.SetProperty("DedicatedUsage", (value))
}

// GetDedicatedUsage gets the value of DedicatedUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) GetPropertyDedicatedUsage() (value uint64, err error) {
	retValue, err := instance.GetProperty("DedicatedUsage")
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

// SetLocalUsage sets the value of LocalUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) SetPropertyLocalUsage(value uint64) (err error) {
	return instance.SetProperty("LocalUsage", (value))
}

// GetLocalUsage gets the value of LocalUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) GetPropertyLocalUsage() (value uint64, err error) {
	retValue, err := instance.GetProperty("LocalUsage")
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

// SetNonLocalUsage sets the value of NonLocalUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) SetPropertyNonLocalUsage(value uint64) (err error) {
	return instance.SetProperty("NonLocalUsage", (value))
}

// GetNonLocalUsage gets the value of NonLocalUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) GetPropertyNonLocalUsage() (value uint64, err error) {
	retValue, err := instance.GetProperty("NonLocalUsage")
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

// SetSharedUsage sets the value of SharedUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) SetPropertySharedUsage(value uint64) (err error) {
	return instance.SetProperty("SharedUsage", (value))
}

// GetSharedUsage gets the value of SharedUsage for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) GetPropertySharedUsage() (value uint64, err error) {
	retValue, err := instance.GetProperty("SharedUsage")
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

// SetTotalCommitted sets the value of TotalCommitted for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) SetPropertyTotalCommitted(value uint64) (err error) {
	return instance.SetProperty("TotalCommitted", (value))
}

// GetTotalCommitted gets the value of TotalCommitted for the instance
func (instance *Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) GetPropertyTotalCommitted() (value uint64, err error) {
	retValue, err := instance.GetProperty("TotalCommitted")
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
