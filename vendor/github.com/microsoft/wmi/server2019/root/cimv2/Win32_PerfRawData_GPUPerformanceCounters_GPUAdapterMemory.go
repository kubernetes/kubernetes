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

// Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory struct
type Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory struct {
	*Win32_PerfRawData

	//
	DedicatedUsage uint64

	//
	SharedUsage uint64

	//
	TotalCommitted uint64
}

func NewWin32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemoryEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemoryEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetDedicatedUsage sets the value of DedicatedUsage for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory) SetPropertyDedicatedUsage(value uint64) (err error) {
	return instance.SetProperty("DedicatedUsage", (value))
}

// GetDedicatedUsage gets the value of DedicatedUsage for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory) GetPropertyDedicatedUsage() (value uint64, err error) {
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

// SetSharedUsage sets the value of SharedUsage for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory) SetPropertySharedUsage(value uint64) (err error) {
	return instance.SetProperty("SharedUsage", (value))
}

// GetSharedUsage gets the value of SharedUsage for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory) GetPropertySharedUsage() (value uint64, err error) {
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
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory) SetPropertyTotalCommitted(value uint64) (err error) {
	return instance.SetProperty("TotalCommitted", (value))
}

// GetTotalCommitted gets the value of TotalCommitted for the instance
func (instance *Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory) GetPropertyTotalCommitted() (value uint64, err error) {
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
